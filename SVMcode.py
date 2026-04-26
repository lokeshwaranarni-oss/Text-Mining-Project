# Support Vector Machine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load Dataset
data = pd.read_csv(
    "SVM_customer churn dataset.csv"
)

# Features and Target
X = data.iloc[:, :-1]

y = data.iloc[:, -1]

# Scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = SVC(
    kernel='rbf',
    C=10,
    probability=True
)

model.fit(
    X_train,
    y_train
)

# Prediction
y_pred = model.predict(
    X_test
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("SVM Accuracy:", accuracy * 100)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ROC Curve (Fixed)
y_prob = model.predict_proba(
    X_test
)[:, 1]

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label="AUC = %.2f" % auc
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.grid()

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# Precision Recall Curve (Fixed)
precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure()

plt.plot(
    recall,
    precision,
    linewidth=2
)

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.grid()

plt.title("Precision-Recall Curve")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.show()

# Feature Importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

importance = result.importances_mean

plt.figure()

sns.barplot(
    x=importance,
    y=data.columns[:-1]
)

plt.title("Feature Importance")

plt.show()

# Accuracy Visualization
plt.figure()

plt.bar(
    ["SVM Accuracy"],
    [accuracy * 100]
)

plt.title("Model Accuracy")

plt.ylabel("Percentage")

plt.show()