"""
Bias Fix for Telemetry Migration Prediction
=============================================
Companion to telemetry_model_training.py — addresses the severe class
imbalance (93% Success / 7% Failed) with a 4-layer correction strategy.

NEW file — does NOT modify any existing files.

What was wrong
--------------
1. Class imbalance    : 27 Success vs 2 Failed (13.5:1 ratio)
2. Trivial test set   : All 6 test samples were class 1 (Success)
3. Threshold too soft : 100 km home-return counted as Success — too generous
4. GREG_5 dominance   : 15 of 29 events are from one bird (data leak risk)

Fixes applied (in order)
-------------------------
Fix 1 — Stricter success labeling
    Re-labels the existing 29 events using a tighter 60 km home-return
    threshold (was 100 km). This converts borderline trips into Failed,
    generating more negative examples from real data.

Fix 2 — Bird-level SMOTE (Synthetic Minority Oversampling)
    Uses SMOTE from imbalanced-learn to synthetically generate Failed
    samples in feature space. Applied ONLY on the training fold to
    prevent data leakage into test folds.

Fix 3 — Remove GREG_5 dominance via per-bird weighting
    GREG_5 contributes 15/29 events (52%). We apply sample_weight in
    training so each bird contributes equally, preventing one bird from
    dominating the model's learned patterns.

Fix 4 — Stratified K-Fold cross-validation (k=5)
    Replaces the single 80/20 split with 5-fold stratified CV. Each fold
    is guaranteed to have both classes in train and test. Reports
    per-fold and mean ± std metrics for honest evaluation.

Output (saved to telemetry_bias_corrected_results/)
----------------------------------------------------
bias_metrics_per_fold.csv      — per-fold Acc/P/R/F1 for all models
bias_metrics_summary.csv       — mean ± std across folds
bias_cv_comparison.png         — grouped bar chart (mean ± std error bars)
bias_class_distribution.png    — before/after class balance visual
bias_detailed_report.txt       — full per-fold classification reports
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not found. Run: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

INPUT_CSV  = "telemetry_migration_events.csv"   # produced by telemetry_preprocess.py
OUTPUT_DIR = "telemetry_bias_corrected_results"

# Tighter home-return threshold for success labeling (was 100 km)
STRICT_HOME_RETURN_KM = 60

# Cross-validation folds
N_FOLDS = 5

# Breeding colony centre used as fallback
BREEDING_LAT = 38.03
BREEDING_LON = -122.75


# ================================================================
# STEP 1 — HAVERSINE (copied to keep file self-contained)
# ================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# ================================================================
# STEP 2 — STRICTER TARGET LABELING  (Fix 1)
# ================================================================

def relabel_strict(df, home_return_km=STRICT_HOME_RETURN_KM):
    """
    Re-label Migration_Success using a tighter home-return radius.

    A migration event is Successful ONLY if:
    - The event end-point is within home_return_km of the bird's
      approximate home base (median lat/lon of all its fixes that
      are already available as start_lat/start_lon of home events), OR
    - The next event for the same bird starts within home_return_km
      (bird demonstrably returned between trips).

    Tightening from 100 km -> 60 km forces borderline events to be
    labeled Failed, creating more negative training examples.

    Parameters
    ----------
    df            : pd.DataFrame   migration events from telemetry_preprocess.py
    home_return_km: float          distance threshold for "returned home"

    Returns
    -------
    pd.DataFrame  : copy with Migration_Success re-labeled
    """
    df = df.copy()
    df["Migration_Success"] = 0

    # Build per-bird home base from the start_lat/start_lon of all their events
    # (these are the locations where the bird was before migrating)
    home_bases = (
        df.groupby("bird_id")[["start_lat", "start_lon"]]
        .median()
        .rename(columns={"start_lat": "home_lat", "start_lon": "home_lon"})
    )

    for bird_id, grp in df.groupby("bird_id"):
        grp = grp.sort_values("start_doy")
        idxs = grp.index.tolist()
        h_lat = home_bases.loc[bird_id, "home_lat"]
        h_lon = home_bases.loc[bird_id, "home_lon"]

        for pos, idx in enumerate(idxs):
            end_lat = df.loc[idx, "end_lat"]
            end_lon = df.loc[idx, "end_lon"]

            # Did the event end near home?
            dist_end_home = haversine_km(end_lat, end_lon, h_lat, h_lon)
            if dist_end_home <= home_return_km:
                df.loc[idx, "Migration_Success"] = 1
                continue

            # Did the NEXT event start near home (confirmed round-trip)?
            if pos + 1 < len(idxs):
                next_idx = idxs[pos + 1]
                ns_lat = df.loc[next_idx, "start_lat"]
                ns_lon = df.loc[next_idx, "start_lon"]
                if haversine_km(ns_lat, ns_lon, h_lat, h_lon) <= home_return_km:
                    df.loc[idx, "Migration_Success"] = 1

    return df


# ================================================================
# STEP 3 — SAMPLE WEIGHTS FOR BIRD EQUITY  (Fix 3)
# ================================================================

def compute_sample_weights(df):
    """
    Assign each event a weight = 1 / (events contributed by that bird).

    Prevents GREG_5 (15 events, 52% of data) from dominating training.
    Each bird contributes equal total weight regardless of how many
    events it produced.

    Returns
    -------
    np.ndarray : weight for each row in df
    """
    bird_counts  = df["bird_id"].map(df["bird_id"].value_counts())
    sample_weights = 1.0 / bird_counts.values
    # Normalise so they sum to len(df)
    sample_weights = sample_weights / sample_weights.mean()
    return sample_weights


# ================================================================
# STEP 4 — BUILD MODELS (same 6 as telemetry_model_training.py)
# ================================================================

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42,
            class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=4, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=6,
            class_weight="balanced"
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0,
            scale_pos_weight=13,   # handles imbalance natively in XGBoost
        )
    return models


# ================================================================
# STEP 5 — STRATIFIED K-FOLD CV WITH SMOTE  (Fixes 2 + 4)
# ================================================================

def run_cross_validation(X, y, sample_weights, n_folds=N_FOLDS):
    """
    Stratified K-Fold CV with per-fold SMOTE oversampling.

    SMOTE is fitted exclusively on the training fold in each iteration
    to prevent information leakage into the test fold.

    Per-fold workflow
    -----------------
    1. Split into train / test fold (stratified)
    2. Scale features (fit on train only)
    3. Apply SMOTE to the training fold only
    4. Fit each model on SMOTE-augmented data + sample weights
    5. Evaluate on the original (unaugmented) test fold
    6. Store per-fold Accuracy, Precision, Recall, F1

    Returns
    -------
    dict : model_name -> list of per-fold metric dicts
    """
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = build_models()

    fold_results = {name: [] for name in models}
    feature_names = list(X.columns)

    print(f"\nRunning {n_folds}-Fold Stratified CV ...\n")
    print(f"{'Fold':<6} {'Model':<22} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6}  "
          f"{'Train(orig)':>12} {'SMOTE->':>8} {'Test':>6}")
    print("-" * 80)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_f = X.iloc[train_idx].copy()
        X_test_f  = X.iloc[test_idx].copy()
        y_train_f = y.iloc[train_idx].copy()
        y_test_f  = y.iloc[test_idx].copy()
        sw_train  = sample_weights[train_idx]

        # --- Scale (fit ONLY on training fold) ---
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(
            scaler.fit_transform(X_train_f),
            columns=feature_names
        )
        X_test_sc = pd.DataFrame(
            scaler.transform(X_test_f),
            columns=feature_names
        )

        # --- SMOTE on training fold only ---
        n_orig_train = len(X_train_sc)
        if SMOTE_AVAILABLE and y_train_f.nunique() >= 2:
            # k_neighbors must be < minority class count
            minority_count = int(y_train_f.value_counts().min())
            k_neighbors    = max(1, min(5, minority_count - 1))
            try:
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors,
                    sampling_strategy="minority"
                )
                X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train_f)
                X_train_sm = pd.DataFrame(X_train_sm, columns=feature_names)
                # Sample weights for synthetic samples = mean of existing weights
                n_synth = len(X_train_sm) - n_orig_train
                sw_synth = np.full(n_synth, sw_train.mean())
                sw_augmented = np.concatenate([sw_train, sw_synth])
            except Exception as e:
                print(f"  [WARN] SMOTE failed fold {fold_idx}: {e}")
                X_train_sm, y_train_sm = X_train_sc, y_train_f
                sw_augmented = sw_train
        else:
            X_train_sm, y_train_sm = X_train_sc, y_train_f
            sw_augmented = sw_train

        n_smote_train = len(X_train_sm)
        n_test = len(X_test_sc)

        # --- Train & evaluate each model ---
        for name, model in models.items():
            mod_clone = type(model)(**model.get_params())

            # Pass sample_weight for models that support it
            fit_kwargs = {}
            if hasattr(mod_clone, "fit"):
                import inspect
                sig = inspect.signature(mod_clone.fit)
                if "sample_weight" in sig.parameters:
                    fit_kwargs["sample_weight"] = sw_augmented

            mod_clone.fit(X_train_sm, y_train_sm, **fit_kwargs)
            y_pred = mod_clone.predict(X_test_sc)

            kw = {"zero_division": 0}
            metrics = {
                "fold":      fold_idx,
                "accuracy":  accuracy_score(y_test_f, y_pred),
                "precision": precision_score(y_test_f, y_pred, **kw),
                "recall":    recall_score(y_test_f, y_pred, **kw),
                "f1":        f1_score(y_test_f, y_pred, **kw),
                "y_test":    y_test_f.values,
                "y_pred":    y_pred,
                "cm":        confusion_matrix(y_test_f, y_pred, labels=[0, 1]),
            }
            fold_results[name].append(metrics)

            if name == list(models.keys())[0]:
                print(
                    f"{'Fold '+str(fold_idx):<6} "
                    f"{name:<22} "
                    f"{metrics['accuracy']:>6.3f} "
                    f"{metrics['precision']:>6.3f} "
                    f"{metrics['recall']:>6.3f} "
                    f"{metrics['f1']:>6.3f}  "
                    f"{n_orig_train:>12} "
                    f"{'->'+str(n_smote_train):>8} "
                    f"{n_test:>6}"
                )
            else:
                print(
                    f"{'':6} {name:<22} "
                    f"{metrics['accuracy']:>6.3f} "
                    f"{metrics['precision']:>6.3f} "
                    f"{metrics['recall']:>6.3f} "
                    f"{metrics['f1']:>6.3f}"
                )
        print()

    return fold_results


# ================================================================
# STEP 6 — SUMMARISE CV RESULTS
# ================================================================

def summarise_cv(fold_results):
    """
    Compute mean and std of each metric across folds for each model.

    Returns
    -------
    pd.DataFrame : one row per model, columns = metric_mean / metric_std
    """
    rows = []
    for name, folds in fold_results.items():
        accs  = [f["accuracy"]  for f in folds]
        precs = [f["precision"] for f in folds]
        recs  = [f["recall"]    for f in folds]
        f1s   = [f["f1"]        for f in folds]
        rows.append({
            "Model":          name,
            "Accuracy_mean":  np.mean(accs),
            "Accuracy_std":   np.std(accs),
            "Precision_mean": np.mean(precs),
            "Precision_std":  np.std(precs),
            "Recall_mean":    np.mean(recs),
            "Recall_std":     np.std(recs),
            "F1_mean":        np.mean(f1s),
            "F1_std":         np.std(f1s),
        })
    return pd.DataFrame(rows)


# ================================================================
# STEP 7 — VISUALISATIONS
# ================================================================

def plot_class_distribution(before_df, after_df):
    """
    Side-by-side bar chart: class distribution before and after bias fix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, title in [
        (axes[0], before_df, "BEFORE: Original Labels\n(100 km return thresh)"),
        (axes[1], after_df,  "AFTER: Strict Labels\n(60 km return thresh)"),
    ]:
        counts = df["Migration_Success"].value_counts().sort_index()
        labels = ["Failed (0)", "Success (1)"]
        colors = ["#e74c3c", "#2ecc71"]
        bars   = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val}\n({val/len(df)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Events", fontsize=11)
        ax.set_ylim(0, max(counts.values) * 1.25)
        ax.grid(axis="y", alpha=0.3)

        ratio = counts.max() / max(counts.min(), 1)
        ax.text(0.5, 0.92, f"Imbalance ratio  {ratio:.1f}:1",
                transform=ax.transAxes, ha="center", fontsize=11,
                color="#c0392b" if ratio > 5 else "#27ae60",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#ffeaa7" if ratio > 5 else "#d5f5e3"))

    plt.suptitle("Class Distribution — Before vs After Bias Fix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_cv_comparison(summary_df):
    """
    Grouped bar chart of Mean Accuracy / Precision / Recall / F1
    with standard-deviation error bars across CV folds.
    """
    model_names = summary_df["Model"].tolist()
    metrics     = ["Accuracy", "Precision", "Recall", "F1"]
    colors      = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    x     = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 7))

    for i, metric in enumerate(metrics):
        means = summary_df[f"{metric}_mean"].values
        stds  = summary_df[f"{metric}_std"].values
        bars  = ax.bar(x + i * width, means, width,
                       label=metric, color=colors[i],
                       yerr=stds, capsize=4, error_kw={"elinewidth": 1.5})
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=45)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (mean ± std across folds)", fontsize=12)
    ax.set_title(
        f"Cross-Validation Results — {N_FOLDS}-Fold Stratified CV\n"
        "with SMOTE + Strict Labeling + Per-Bird Sample Weights",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.25)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1,
               label="70% target line")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_cv_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_f1_boxplot(fold_results):
    """
    Box-plot of F1 scores across folds per model — shows variance.
    """
    model_names = list(fold_results.keys())
    f1_data     = [[f["f1"] for f in fold_results[m]] for m in model_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(f1_data, labels=model_names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"F1 Score Distribution Across {N_FOLDS} Folds",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.7, color="red", linestyle="--", linewidth=1.5,
               label="70% target")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_f1_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# STEP 8 — SAVE REPORTS
# ================================================================

def save_per_fold_csv(fold_results, summary_df):
    """Save per-fold metrics and mean-std summary to CSV."""
    rows = []
    for name, folds in fold_results.items():
        for f in folds:
            rows.append({
                "Model":     name,
                "Fold":      f["fold"],
                "Accuracy":  f["accuracy"],
                "Precision": f["precision"],
                "Recall":    f["recall"],
                "F1":        f["f1"],
            })

    per_fold_df = pd.DataFrame(rows)
    p1 = os.path.join(OUTPUT_DIR, "bias_metrics_per_fold.csv")
    p2 = os.path.join(OUTPUT_DIR, "bias_metrics_summary.csv")
    per_fold_df.to_csv(p1, index=False)
    summary_df.to_csv(p2, index=False)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


def save_bias_report(fold_results, summary_df, before_df, after_df):
    """Write a human-readable text report."""
    path = os.path.join(OUTPUT_DIR, "bias_detailed_report.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BIAS CORRECTION REPORT -- TELEMETRY MIGRATION DATA\n")
        f.write("=" * 70 + "\n\n")

        # --- Class balance ---
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        b_counts = before_df["Migration_Success"].value_counts().sort_index()
        a_counts = after_df["Migration_Success"].value_counts().sort_index()
        f.write(f"  BEFORE (100 km thresh):  "
                f"Failed={b_counts.get(0,0)}  "
                f"Success={b_counts.get(1,0)}  "
                f"Ratio={b_counts.max()/max(b_counts.min(),1):.1f}:1\n")
        f.write(f"  AFTER  ( 60 km thresh):  "
                f"Failed={a_counts.get(0,0)}  "
                f"Success={a_counts.get(1,0)}  "
                f"Ratio={a_counts.max()/max(a_counts.min(),1):.1f}:1\n\n")

        # --- Strategy summary ---
        f.write("BIAS CORRECTION STRATEGY\n")
        f.write("-" * 40 + "\n")
        f.write("  Fix 1: Stricter target labeling (100 km -> 60 km home-return radius)\n")
        f.write("  Fix 2: SMOTE synthetic minority oversampling (training folds only)\n")
        f.write("  Fix 3: Per-bird sample weights (GREG_5 downweighted)\n")
        f.write(f"  Fix 4: {N_FOLDS}-Fold Stratified Cross-Validation\n\n")

        # --- CV summary table ---
        f.write("CROSS-VALIDATION SUMMARY (mean +/- std)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Model':<22} {'Acc':>12} {'Precision':>12} "
                f"{'Recall':>12} {'F1':>12}\n")
        f.write("  " + "-" * 60 + "\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"  {row['Model']:<22} "
                f"{row['Accuracy_mean']:.3f}+/-{row['Accuracy_std']:.3f}  "
                f"{row['Precision_mean']:.3f}+/-{row['Precision_std']:.3f}  "
                f"{row['Recall_mean']:.3f}+/-{row['Recall_std']:.3f}  "
                f"{row['F1_mean']:.3f}+/-{row['F1_std']:.3f}\n"
            )
        f.write("\n")

        # --- Per-fold details ---
        f.write("PER-FOLD DETAILS\n")
        f.write("-" * 70 + "\n")
        for name, folds in fold_results.items():
            f.write(f"\n{name}\n")
            for fold in folds:
                f.write(
                    f"  Fold {fold['fold']}: "
                    f"Acc={fold['accuracy']:.3f}  "
                    f"P={fold['precision']:.3f}  "
                    f"R={fold['recall']:.3f}  "
                    f"F1={fold['f1']:.3f}\n"
                )
                cm = fold["cm"]
                f.write(f"    CM: TN={cm[0,0]} FP={cm[0,1]} "
                        f"FN={cm[1,0]} TP={cm[1,1]}\n")
                f.write("    " + classification_report(
                    fold["y_test"], fold["y_pred"],
                    labels=[0, 1],
                    target_names=["Failed", "Success"],
                    zero_division=0
                ).replace("\n", "\n    ") + "\n")

    print(f"Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load existing aggregated events ----
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"'{INPUT_CSV}' not found.\n"
            "Run telemetry_preprocess.py first:\n"
            "  python telemetry_preprocess.py"
        )

    print("=" * 65)
    print(" TELEMETRY BIAS CORRECTION PIPELINE")
    print("=" * 65)

    before_df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded: {len(before_df)} migration events")
    orig_counts = before_df["Migration_Success"].value_counts().sort_index()
    print(f"Original class dist  : Failed={orig_counts.get(0,0)}  "
          f"Success={orig_counts.get(1,0)}  "
          f"Ratio={orig_counts.max()/max(orig_counts.min(),1):.1f}:1")

    # ---- Fix 1: Strict re-labeling ----
    print("\n[Fix 1] Applying strict home-return labeling "
          f"({STRICT_HOME_RETURN_KM} km threshold) ...")
    after_df = relabel_strict(before_df, home_return_km=STRICT_HOME_RETURN_KM)
    new_counts = after_df["Migration_Success"].value_counts().sort_index()
    print(f"New class dist       : Failed={new_counts.get(0,0)}  "
          f"Success={new_counts.get(1,0)}  "
          f"Ratio={new_counts.max()/max(new_counts.min(),1):.1f}:1")

    # ---- Prepare feature matrix ----
    drop_cols = ["bird_id", "event_id", "Migration_Success"]
    X = after_df.drop(columns=drop_cols, errors="ignore")
    y = after_df["Migration_Success"]

    # ---- Fix 3: Per-bird sample weights ----
    print("\n[Fix 3] Computing per-bird sample weights ...")
    sample_weights = compute_sample_weights(after_df)
    bird_contributions = after_df.groupby("bird_id").size()
    print("  Events per bird:")
    for bird, count in bird_contributions.items():
        w = sample_weights[after_df["bird_id"] == bird].mean()
        print(f"    {bird:<10} {count:>3} events  weight={w:.3f}")

    # ---- Impute missing values (median) ----
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---- Check if CV is possible ----
    global N_FOLDS
    min_class = new_counts.min()
    if min_class < N_FOLDS:
        actual_folds = max(2, int(min_class))
        print(f"\n[WARN] Minority class has {min_class} samples. "
              f"Reducing to {actual_folds}-fold CV.")
    else:
        actual_folds = N_FOLDS

    # ---- Fix 2 + Fix 4: SMOTE + Stratified K-Fold CV ----
    if SMOTE_AVAILABLE:
        print(f"\n[Fix 2] SMOTE enabled (applied per-fold on training data only)")
    else:
        print("\n[Fix 2] SMOTE not available -- skipping oversampling")

    N_FOLDS = actual_folds

    fold_results = run_cross_validation(X, y, sample_weights, n_folds=actual_folds)

    # ---- Summarise ----
    summary_df = summarise_cv(fold_results)

    print("\n" + "=" * 65)
    print(" CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    print(f"\n{'Model':<22} {'Accuracy':>12} {'Precision':>12} "
          f"{'Recall':>12} {'F1':>12}")
    print("-" * 62)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<22} "
            f"{row['Accuracy_mean']:.3f}+/-{row['Accuracy_std']:.2f}  "
            f"{row['Precision_mean']:.3f}+/-{row['Precision_std']:.2f}  "
            f"{row['Recall_mean']:.3f}+/-{row['Recall_std']:.2f}  "
            f"{row['F1_mean']:.3f}+/-{row['F1_std']:.2f}"
        )

    # ---- Plots ----
    print("\n" + "=" * 65)
    print(" GENERATING PLOTS")
    print("=" * 65 + "\n")

    plot_class_distribution(before_df, after_df)
    plot_cv_comparison(summary_df)
    plot_f1_boxplot(fold_results)

    # ---- Save reports ----
    save_per_fold_csv(fold_results, summary_df)
    save_bias_report(fold_results, summary_df, before_df, after_df)

    print("\n" + "=" * 65)
    print(f" ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
