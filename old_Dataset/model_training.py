"""
Model Training and Evaluation on Full Bucket Data

Trains 5 models and generates performance metrics with visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, precision_recall_curve)

OUTPUT_DIR = "results"


def load_full_data():
    """Load FULL bucket train and test data."""
    train_df = pd.read_csv(f"{OUTPUT_DIR}/../bucket_data/full_train.csv")
    test_df = pd.read_csv(f"{OUTPUT_DIR}/../bucket_data/full_test.csv")
    
    X_train = train_df.drop(columns=["Migration_Success"])
    X_test = test_df.drop(columns=["Migration_Success"])
    y_train = train_df["Migration_Success"]
    y_test = test_df["Migration_Success"]
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Train all models and return predictions."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    
    print("="*70)
    print("TRAINING MODELS ON FULL DATA")
    print("="*70)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print()
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": y_test,
            "metrics": {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
        }
        
        print(f"  Accuracy:  {results[name]['metrics']['accuracy']:.4f}")
        print(f"  Precision: {results[name]['metrics']['precision']:.4f}")
        print(f"  Recall:    {results[name]['metrics']['recall']:.4f}")
        print(f"  F1 Score: {results[name]['metrics']['f1']:.4f}")
        print()
    
    return results


def create_metrics_comparison_plot(results):
    """Create bar chart comparing metrics across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    
    for i, metric in enumerate(metrics):
        values = [results[m]["metrics"][metric] for m in model_names]
        bars = ax.bar(x + i*width, values, width, label=metric.capitalize(), color=colors[i])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/metrics_comparison.png")


def create_confusion_matrices(results):
    """Create subplot of confusion matrices for all models."""
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result["metrics"]["confusion_matrix"]
        
        im = axes[idx].imshow(cm, cmap='Blues')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Failed', 'Success'])
        axes[idx].set_yticklabels(['Failed', 'Success'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', 
                              fontsize=16, fontweight='bold',
                              color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        total = cm.sum()
        accuracy = (cm[0,0] + cm[1,1]) / total
        axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/confusion_matrices.png")


def create_roc_curves(results):
    """Create ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        if result["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(result["y_test"], result["y_prob"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/roc_curves.png")


def create_precision_recall_curves(results):
    """Create Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        if result["y_prob"] is not None:
            precision, recall, _ = precision_recall_curve(result["y_test"], result["y_prob"])
            ax.plot(recall, precision, linewidth=2, label=name)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/precision_recall_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/precision_recall_curves.png")


def create_model_accuracy_ranking(results):
    """Create horizontal bar chart ranking models by accuracy."""
    model_names = list(results.keys())
    accuracies = [results[m]["metrics"]["accuracy"] for m in model_names]
    
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_acc = [accuracies[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_acc, color=colors)
    
    for bar, acc in zip(bars, sorted_acc):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{acc:.4f}', va='center', fontsize=11)
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Model Ranking by Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(sorted_acc) * 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/accuracy_ranking.png")


def create_f1_score_comparison(results):
    """Create grouped bar chart for F1 score comparison."""
    model_names = list(results.keys())
    
    precision = [results[m]["metrics"]["precision"] for m in model_names]
    recall = [results[m]["metrics"]["recall"] for m in model_names]
    f1 = [results[m]["metrics"]["f1"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, and F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/f1_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/f1_comparison.png")


def save_detailed_report(results, y_test):
    """Save detailed classification reports to text file."""
    with open(f"{OUTPUT_DIR}/detailed_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT - FULL DATA\n")
        f.write("="*70 + "\n\n")
        
        for name, result in results.items():
            f.write("-"*70 + "\n")
            f.write(f"MODEL: {name}\n")
            f.write("-"*70 + "\n\n")
            
            f.write("METRICS:\n")
            f.write(f"  Accuracy:  {result['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {result['metrics']['recall']:.4f}\n")
            f.write(f"  F1 Score: {result['metrics']['f1']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            cm = result['metrics']['confusion_matrix']
            f.write(f"                Predicted\n")
            f.write(f"              Failed  Success\n")
            f.write(f"Actual Failed   {cm[0,0]:4d}    {cm[0,1]:4d}\n")
            f.write(f"       Success {cm[1,0]:4d}    {cm[1,1]:4d}\n\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write(classification_report(y_test, result["y_pred"], 
                                         target_names=['Failed', 'Success']))
            f.write("\n")
    
    print(f"Saved: {OUTPUT_DIR}/detailed_report.txt")


def save_metrics_csv(results):
    """Save metrics summary as CSV."""
    rows = []
    for name, result in results.items():
        cm = result['metrics']['confusion_matrix']
        rows.append({
            "Model": name,
            "Accuracy": result['metrics']['accuracy'],
            "Precision": result['metrics']['precision'],
            "Recall": result['metrics']['recall'],
            "F1_Score": result['metrics']['f1'],
            "TN": cm[0,0],
            "FP": cm[0,1],
            "FN": cm[1,0],
            "TP": cm[1,1]
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/metrics_summary.csv")
    print("\n" + df.to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_full_data()
    
    results = train_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    create_metrics_comparison_plot(results)
    create_confusion_matrices(results)
    create_roc_curves(results)
    create_precision_recall_curves(results)
    create_model_accuracy_ranking(results)
    create_f1_score_comparison(results)
    
    save_detailed_report(results, y_test)
    save_metrics_csv(results)
    
    print("\n" + "="*70)
    print("ALL RESULTS SAVED TO:", OUTPUT_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
