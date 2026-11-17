"""
Decision Tree classifier on the Breast Cancer dataset.
Run: python decision_tree.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer()
    return (data.data, data.target,
            list(data.target_names), list(data.feature_names))  # 0=malignant, 1=benign


def plot_depth_vs_accuracy(X_train, y_train, X_test, y_test, criterion, max_depth=15):
    """Train/test accuracy for different max_depth values (shows overfitting)."""
    depths = range(1, max_depth + 1)
    train_scores, test_scores = [], []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=criterion,
                                      random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        train_scores.append(tree.score(X_train, y_train))
        test_scores.append(tree.score(X_test, y_test))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, marker="o", label="Train accuracy")
    plt.plot(depths, test_scores, marker="s", label="Test accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Overfitting: max_depth vs accuracy (Breast Cancer)")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIG_DIR / "depth_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree Confusion Matrix")
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Decision Tree ROC Curve")
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tree_diagram(model, feature_names, class_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=8)
    plt.title(f"Decision Tree (criterion={model.criterion}, max_depth={model.max_depth})")
    plt.savefig(FIG_DIR / "tree_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[top_idx], y=[feature_names[i] for i in top_idx],
                color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (Decision Tree)")
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nTop 5 features:")
    for i in top_idx[:5]:
        print(f"  {feature_names[i]:<25} {importances[i]:.4f}")


def main():
    X, y, class_names, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline: a tree with no depth limit memorises the training data.
    baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    print("Baseline tree (no max_depth)")
    print(f"  Depth: {baseline.get_depth()}")
    print(f"  Train accuracy: {baseline.score(X_train, y_train):.4f}")
    print(f"  Test accuracy:  {baseline.score(X_test, y_test):.4f}\n")

    # Unlike KNN, trees split on thresholds, so no scaling (and no Pipeline) is needed.
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
    }

    grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                        param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Gini vs entropy: best CV accuracy for each criterion
    results = grid.cv_results_
    for criterion in ["gini", "entropy"]:
        mask = np.array(results["param_criterion"]) == criterion
        print(f"Best CV accuracy with {criterion}: {results['mean_test_score'][mask].max():.4f}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # probability of benign

    print(f"\nTrain accuracy: {best_model.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification report:\n",
          classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):\n", cm)
    print(f"Malignant predicted as benign (false negatives): {cm[0, 1]}")
    print(f"Benign predicted as malignant (false alarms): {cm[1, 0]}")

    plot_depth_vs_accuracy(X_train, y_train, X_test, y_test, best_model.criterion)
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curve(best_model, X_test, y_test)
    plot_tree_diagram(best_model, feature_names, class_names)
    plot_feature_importance(best_model, feature_names)

    joblib.dump(best_model, BASE_DIR / "model.pkl")
    print("\nSaved model.pkl and figures/")

    # Quick prediction demo using one real test sample
    sample = X_test[:1]
    print("Sample prediction:", class_names[best_model.predict(sample)[0]],
          "| Actual:", class_names[y_test[0]])


if __name__ == "__main__":
    main()