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
            y_prob = (
                mod_clone.predict_proba(X_test_sc)[:, 1]
                if hasattr(mod_clone, "predict_proba")
                else None
            )

            kw = {"zero_division": 0}
            metrics = {
                "fold":      fold_idx,
                "accuracy":  accuracy_score(y_test_f, y_pred),
                "precision": precision_score(y_test_f, y_pred, **kw),
                "recall":    recall_score(y_test_f, y_pred, **kw),
                "f1":        f1_score(y_test_f, y_pred, **kw),
                "y_test":    y_test_f.values,
                "y_pred":    y_pred,
                "y_prob":    y_prob,
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
# STEP 8 — ADDITIONAL VISUALISATIONS (mirrors telemetry_model_training.py)
# ================================================================

def plot_confusion_matrices(fold_results):
    """
    Aggregate confusion matrices across all CV folds per model and
    display as a grid — mirrors create_confusion_matrices() in
    telemetry_model_training.py.
    """
    from sklearn.metrics import roc_curve, auc   # lazy import to keep top clean
    model_names = list(fold_results.keys())
    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        # Sum CMs across folds
        cm = np.zeros((2, 2), dtype=int)
        for fold in fold_results[name]:
            cm += fold["cm"]

        im = axes[idx].imshow(cm, cmap="Blues")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(["Failed", "Success"])
        axes[idx].set_yticklabels(["Failed", "Success"])
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_title(name, fontsize=12, fontweight="bold")

        for i in range(2):
            for j in range(2):
                axes[idx].text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        total = cm.sum()
        accuracy = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0
        axes[idx].text(
            0.5, -0.15, f"Accuracy (pooled): {accuracy:.2%}",
            transform=axes[idx].transAxes,
            ha="center", fontsize=10,
        )

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Confusion Matrices — All Models (Bias-Corrected, Pooled CV Folds)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(fold_results):
    """
    Pool y_test / y_prob across all CV folds and draw one ROC curve per
    model — mirrors create_roc_curves() in telemetry_model_training.py.
    """
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, folds in fold_results.items():
        y_test_all = np.concatenate([f["y_test"] for f in folds])
        probs = [f["y_prob"] for f in folds if f["y_prob"] is not None]
        if not probs:
            continue
        y_prob_all = np.concatenate(probs)
        fpr, tpr, _ = roc_curve(y_test_all, y_prob_all)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves — All Models (Bias-Corrected, Pooled CV Folds)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_precision_recall_curves(fold_results):
    """
    Pool y_test / y_prob across CV folds and draw one PR curve per model
    — mirrors create_precision_recall_curves() in telemetry_model_training.py.
    """
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, folds in fold_results.items():
        y_test_all = np.concatenate([f["y_test"] for f in folds])
        probs = [f["y_prob"] for f in folds if f["y_prob"] is not None]
        if not probs:
            continue
        y_prob_all = np.concatenate(probs)
        precision, recall, _ = precision_recall_curve(y_test_all, y_prob_all)
        ax.plot(recall, precision, linewidth=2, label=name)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — All Models (Bias-Corrected, Pooled CV Folds)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_precision_recall_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_ranking(summary_df):
    """
    Horizontal bar chart of mean accuracy per model, sorted ascending
    — mirrors create_model_accuracy_ranking() in telemetry_model_training.py.
    """
    model_names = summary_df["Model"].tolist()
    accuracies  = summary_df["Accuracy_mean"].tolist()

    sorted_idx   = np.argsort(accuracies)
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_acc   = [accuracies[i]  for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars   = ax.barh(sorted_names, sorted_acc, color=colors)

    for bar, acc in zip(bars, sorted_acc):
        ax.text(
            acc + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}", va="center", fontsize=11,
        )

    ax.set_xlabel("Accuracy (CV mean)", fontsize=12)
    ax.set_title(
        "Model Ranking by Accuracy — Bias-Corrected Telemetry Data",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, max(sorted_acc) * 1.15 if sorted_acc else 1)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_accuracy_ranking.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_f1_comparison(summary_df):
    """
    Bar chart of mean F1 Score per model with ±std error bars —
    one centered bar per model, cleanly labeled.
    """
    model_names = summary_df["Model"].tolist()
    f1_means    = summary_df["F1_mean"].tolist()
    f1_stds     = summary_df["F1_std"].tolist()

    x     = np.arange(len(model_names))
    width = 0.5
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e67e22", "#1b5e20", "#e74c3c"]
    bars = ax.bar(
        x, f1_means, width,
        color=colors[:len(model_names)],
        yerr=f1_stds,
        error_kw={"elinewidth": 1.8, "capsize": 6},
        label="F1 Score (mean ± std)",
    )

    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.012,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1 Score (CV mean ± std)", fontsize=12)
    ax.set_title(
        "F1 Score Comparison — Bias-Corrected Telemetry Data",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="0.90 reference")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(X, y, sample_weights, feature_names):
    """
    Train Decision Tree and Random Forest on the full re-labeled dataset
    (with SMOTE if available) and plot their feature importances side-by-side
    — mirrors create_feature_importance_plot() in telemetry_model_training.py.
    """
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # Apply SMOTE if available
    X_fit, y_fit = X_scaled, y
    if SMOTE_AVAILABLE and y.nunique() >= 2:
        minority_count = int(y.value_counts().min())
        k_neighbors    = max(1, min(5, minority_count - 1))
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors,
                          sampling_strategy="minority")
            X_sm, y_sm = smote.fit_resample(X_scaled, y)
            X_fit = pd.DataFrame(X_sm, columns=feature_names)
            y_fit = y_sm
        except Exception:
            pass

    tree_models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=4, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=6,
            class_weight="balanced"
        ),
    }

    fitted = {}
    for name, model in tree_models.items():
        model.fit(X_fit, y_fit)
        fitted[name] = model

    n_plots = len(fitted)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, fitted.items()):
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)[-20:]
        top_names   = [feature_names[i] for i in sorted_idx]
        top_values  = importances[sorted_idx]

        colors = plt.cm.viridis(top_values / top_values.max())
        ax.barh(top_names, top_values, color=colors)
        ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
        ax.set_title(f"{name} — Top 20 Features", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Feature Importances — Bias-Corrected Telemetry Data",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_xgboost_importance(X, y, feature_names):
    """
    Train XGBoost on the full re-labeled dataset and plot its top-20
    feature importances — mirrors create_xgboost_importance_plot() in
    telemetry_model_training.py.
    """
    if not XGBOOST_AVAILABLE:
        return
    from sklearn.preprocessing import StandardScaler

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_fit, y_fit = X_scaled, y
    if SMOTE_AVAILABLE and y.nunique() >= 2:
        minority_count = int(y.value_counts().min())
        k_neighbors    = max(1, min(5, minority_count - 1))
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors,
                          sampling_strategy="minority")
            X_sm, y_sm = smote.fit_resample(X_scaled, y)
            X_fit = pd.DataFrame(X_sm, columns=feature_names)
            y_fit = y_sm
        except Exception:
            pass

    model = XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric="logloss", verbosity=0,
        scale_pos_weight=13,
    )
    model.fit(X_fit, y_fit)

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[-20:]
    top_names   = [feature_names[i] for i in sorted_idx]
    top_values  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(top_values / top_values.max())
    ax.barh(top_names, top_values, color=colors)
    ax.set_xlabel("Feature Importance (F-score)", fontsize=11)
    ax.set_title(
        "XGBoost — Top 20 Feature Importances (Bias-Corrected Telemetry)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_feature_importance_xgb.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_line_graph(fold_results):
    """
    Line graph of accuracy per CV fold for every model.

    Each model gets its own line, so the reader can see per-fold
    trend and variance at a glance — complements the bar charts
    that only show means.
    """
    model_names = list(fold_results.keys())
    # Palette: one colour per model
    palette = [
        "#3498db", "#e74c3c", "#2ecc71",
        "#9b59b6", "#f39c12", "#1abc9c",
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    offset_step = 0.04
    n_models = len(model_names)
    start_offset = - (n_models - 1) / 2.0 * offset_step

    for i, (color, name) in enumerate(zip(palette, model_names)):
        folds      = fold_results[name]
        fold_nums  = [f["fold"] for f in folds]
        accuracies = [f["accuracy"] for f in folds]

        # Add horizontal jitter to prevent perfectly overlapping lines
        jittered_folds = [f_num + start_offset + (i * offset_step) for f_num in fold_nums]

        ax.plot(
            jittered_folds, accuracies,
            marker="o", linewidth=2.2, markersize=7,
            label=name, color=color, alpha=0.85
        )
        # Annotate last point
        ax.annotate(
            f"{accuracies[-1]:.3f}",
            xy=(jittered_folds[-1], accuracies[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=color, va="center", fontweight="bold"
        )

    n_folds = max(len(fold_results[m]) for m in model_names)
    ax.set_xticks(range(1, n_folds + 1))
    ax.set_xticklabels([f"Fold {i}" for i in range(1, n_folds + 1)], fontsize=11)
    ax.set_xlabel("Cross-Validation Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Per-Fold Accuracy — All Models (Bias-Corrected Telemetry Data)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0.8, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% reference")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_accuracy_line_graph.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_data_split_pie(after_df, n_folds):
    """
    Dual-panel pie chart that communicates the data composition:

    Left  — Class distribution after bias-correction relabeling
            (Failed vs Success proportion)
    Right — Approximate train / test split for one CV fold
            (one fold kept as test; rest as training — shown as %)
    """
    counts   = after_df["Migration_Success"].value_counts().sort_index()
    n_failed  = int(counts.get(0, 0))
    n_success = int(counts.get(1, 0))
    total     = n_failed + n_success

    # Approximate train/test split for one fold
    test_size  = round(total / n_folds)
    train_size = total - test_size

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Data Split Overview — Bias-Corrected Telemetry Dataset",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # ---------- Panel 1: Class distribution ----------
    ax_class = axes[0]
    class_sizes  = [n_failed, n_success]
    class_labels = [
        f"Failed\n({n_failed} events, {n_failed/total*100:.1f}%)",
        f"Success\n({n_success} events, {n_success/total*100:.1f}%)",
    ]
    class_colors = ["#e74c3c", "#2ecc71"]
    class_explode = (0.05, 0)

    wedges, _, autotexts = ax_class.pie(
        class_sizes,
        labels=class_labels,
        colors=class_colors,
        explode=class_explode,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax_class.set_title(
        f"Class Distribution\n(after 60 km strict relabeling, n={total})",
        fontsize=12, fontweight="bold",
    )

    # ---------- Panel 2: Train / Test split ----------
    ax_split = axes[1]
    split_sizes  = [train_size, test_size]
    split_labels = [
        f"Training\n({train_size} samples, {train_size/total*100:.1f}%)",
        f"Test (1 fold)\n({test_size} samples, {test_size/total*100:.1f}%)",
    ]
    split_colors  = ["#3498db", "#f39c12"]
    split_explode = (0, 0.07)

    wedges2, _, autotexts2 = ax_split.pie(
        split_sizes,
        labels=split_labels,
        colors=split_colors,
        explode=split_explode,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts2:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax_split.set_title(
        f"Train / Test Split\n({n_folds}-Fold Stratified CV, one fold shown)",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_data_split_pie.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# STEP 9 — SAVE REPORTS
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
    plot_confusion_matrices(fold_results)
    plot_roc_curves(fold_results)
    plot_precision_recall_curves(fold_results)
    plot_accuracy_ranking(summary_df)
    plot_f1_comparison(summary_df)
    feature_names = list(X.columns)
    plot_feature_importance(X, y, sample_weights, feature_names)
    plot_xgboost_importance(X, y, feature_names)
    plot_accuracy_line_graph(fold_results)
    plot_data_split_pie(after_df, N_FOLDS)

    # ---- Save reports ----
    save_per_fold_csv(fold_results, summary_df)
    save_bias_report(fold_results, summary_df, before_df, after_df)

    print("\n" + "=" * 65)
    print(f" ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
