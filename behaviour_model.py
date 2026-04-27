import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# This model analyzes bird behavior patterns during migration
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Scaling
from sklearn.preprocessing import StandardScaler

# ROC (optional)
from sklearn.metrics import roc_curve, auc

# -------------------------------
# LOAD DATA
# -------------------------------
train = pd.read_csv("bucket_data/behaviour_train.csv")
test = pd.read_csv("bucket_data/behaviour_test.csv")

# -------------------------------
# SPLIT FEATURES & TARGET
# -------------------------------
X_train = train.drop("Migration_Success", axis=1)
y_train = train["Migration_Success"]

X_test = test.drop("Migration_Success", axis=1)
y_test = test["Migration_Success"]

# -------------------------------
# SCALING (for KNN & SVM)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# FUNCTION TO TRAIN & EVALUATE
# -------------------------------
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    print("\n" + "="*50)
    print(name)
    print("="*50)

    # Train
    model.fit(X_tr, y_tr)

    # Predict
    y_pred = model.predict(X_te)

    # Metrics
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model

# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model = evaluate_model("Logistic Regression", log_model, X_train, y_train, X_test, y_test)

# -------------------------------
# ROC CURVE (ONLY FOR LOGISTIC)
# -------------------------------
y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Logistic Regression)")
plt.legend()
plt.show()

# -------------------------------
# KNN
# -------------------------------
evaluate_model("KNN", KNeighborsClassifier(n_neighbors=5),
               X_train_scaled, y_train, X_test_scaled, y_test)

# -------------------------------
# SVM
# -------------------------------
evaluate_model("SVM", SVC(kernel="rbf"),
               X_train_scaled, y_train, X_test_scaled, y_test)

# -------------------------------
# DECISION TREE
# -------------------------------
evaluate_model("Decision Tree", DecisionTreeClassifier(),
               X_train, y_train, X_test, y_test)

# -------------------------------
# FINAL CONCLUSION
# -------------------------------
print("\nFINAL CONCLUSION:")
print("All models perform close to ~50% accuracy.")
print("This indicates weak patterns in the dataset.")
print("Logistic Regression performs best among all models.")