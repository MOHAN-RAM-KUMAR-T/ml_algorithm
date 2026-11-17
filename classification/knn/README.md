# K-Nearest Neighbors (KNN) Classifier

Binary classification (malignant vs. benign) on the Breast Cancer Wisconsin dataset.

## What I learned
- How KNN classifies using the labels of the k nearest points
- Why feature scaling is essential for distance-based models
- How k affects bias vs. variance (small k overfits, large k underfits)
- Tuning with GridSearchCV inside a Pipeline to avoid data leakage

## Run
pip install -r ../../requirements.txt
python knn.py

## Results
- Test accuracy: 97%
- Malignant: precision 1.00, recall 0.93
- Benign: precision 0.96, recall 1.00
- ROC-AUC: (paste from terminal)
- Best params: (paste from terminal)

## Observation
All 3 misclassifications were malignant tumors predicted as benign.
In medical screening, false negatives are the costliest error, so a
natural next step is optimizing for malignant recall (e.g. scoring
GridSearchCV with `recall` or adjusting the decision threshold).

## Figures
![Confusion matrix](figures/confusion_matrix.png)
![k comparison](figures/k_value_comparison.png)
![ROC curve](figures/roc_curve.png)