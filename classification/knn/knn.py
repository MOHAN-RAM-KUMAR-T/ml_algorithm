"""
K-Nearest Neighbors classifier on the Breast Cancer dataset.
Run: python knn.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, list(data.target_names)  # 0=malignant, 1=benign


def plot_k_comparison(X_train, y_train, max_k=30):
    """Cross-validated accuracy for different k values."""
    ks = range(1, max_k + 1)
    scores = []
    for k in ks:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=k))])
        scores.append(cross_val_score(pipe, X_train, y_train, cv=5).mean())

    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("5-fold CV accuracy")
    plt.title("Effect of k on KNN accuracy (Breast Cancer)")
    plt.grid(True)
    plt.savefig(FIG_DIR / "k_value_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("KNN Confusion Matrix")
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("KNN ROC Curve")
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X, y, class_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # KNN is distance-based, so scaling is essential.
    # A Pipeline fits the scaler only on training folds (no data leakage).
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    param_grid = {
        "knn__n_neighbors": list(range(1, 21)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # probability of benign

    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification report:\n",
          classification_report(y_test, y_pred, target_names=class_names))

    plot_k_comparison(X_train, y_train)
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curve(best_model, X_test, y_test)

    joblib.dump(best_model, BASE_DIR / "model.pkl")
    print("Saved model.pkl and figures/")

    # Quick prediction demo using one real test sample
    sample = X_test[:1]
    print("Sample prediction:", class_names[best_model.predict(sample)[0]],
          "| Actual:", class_names[y_test[0]])


if __name__ == "__main__":
    main()