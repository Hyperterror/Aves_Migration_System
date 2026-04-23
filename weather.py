# ============================================
# IMPORT LIBRARIES
# ============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)


# ============================================
# LOAD DATA
# ============================================
print("Loading datasets...")

train_df = pd.read_csv("bucket_data/weather_train.csv")
test_df = pd.read_csv("bucket_data/weather_test.csv")

target = "Migration_Success"


# ============================================
# FEATURE ENGINEERING (IMPORTANT)
# ============================================
train_df["Weather_Humidity"] = train_df["Humidity_%"] * train_df["Weather_Condition_Stormy"]
test_df["Weather_Humidity"] = test_df["Humidity_%"] * test_df["Weather_Condition_Stormy"]

train_df["Pressure_Visibility"] = train_df["Pressure_hPa"] / (train_df["Visibility_km"] + 1)
test_df["Pressure_Visibility"] = test_df["Pressure_hPa"] / (test_df["Visibility_km"] + 1)


# ============================================
# BASIC ANALYSIS
# ============================================
print("\nClass Distribution:")
print(train_df[target].value_counts())

print("\nBaseline Accuracy:")
print(train_df[target].value_counts(normalize=True).max())

print("\nCorrelation with target:")
corr = train_df.corr()[target].sort_values(ascending=False)
print(corr)


# ============================================
# FEATURE SELECTION (REMOVE WEAK FEATURES)
# ============================================
important_features = corr[abs(corr) > 0.01].index
print("\nSelected Features:", important_features)

train_df = train_df[important_features]
test_df = test_df[important_features]


# ============================================
# SPLIT DATA
# ============================================
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]


# ============================================
# SCALING
# ============================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================
# MODELS (IMPROVED)
# ============================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel='rbf', C=1, probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}


# ============================================
# FUNCTIONS
# ============================================
def plot_cm(cm, title):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc(y_test, y_score, name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

    return roc_auc


# ============================================
# TRAIN & EVALUATE
# ============================================
results = []

for name, model in models.items():
    print("\n" + "="*50)
    print(name)
    print("="*50)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    plot_cm(cm, f"{name} Confusion Matrix")

    auc_score = plot_roc(y_test, y_score, name)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc_score
    })


# ============================================
# COMPARISON TABLE
# ============================================
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(results_df.sort_values(by="F1 Score", ascending=False))


# ============================================
# BAR GRAPH
# ============================================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Model Comparison (F1 Score)")
plt.xticks(rotation=20)
plt.show()


# ============================================
# BEST MODEL
# ============================================
best = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]

print("\nBEST MODEL:")
print(best)