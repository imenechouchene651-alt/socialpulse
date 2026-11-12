import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

LABELS = ["Positive", "Negative", "Neutral", "Mixed"]

df = pd.read_csv("data/labeled_comments.csv")
df = df.dropna(subset=["comment_text", "sentiment"])
df = df[df["sentiment"].isin(LABELS)]
X, y = df["comment_text"].astype(str), df["sentiment"]

# Même découpage que train_model.py
counts = y.value_counts()
n_classes = y.nunique()
n_test = max(int(round(len(df) * 0.20)), n_classes)
can_stratify = (counts.min() >= 2) and (len(df) - n_test >= n_classes)
if can_stratify:
    _, X_test, _, y_test = train_test_split(X, y, test_size=n_test, random_state=42, stratify=y)
else:
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

os.makedirs("reports", exist_ok=True)
rows = []

for name, file in [("Logistic Regression", "logistic_regression"),
                   ("Linear SVM", "linear_svm")]:
    model = joblib.load(f"models/{file}.joblib")
    pred = model.predict(X_test)

    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    })

    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"reports/confusion_matrix_{file}.png", dpi=150)
    plt.close()

metrics = pd.DataFrame(rows).round(3)
metrics.to_csv("reports/model_comparison.csv", index=False)
print("\n", metrics)
print("\nMeilleur modèle (Macro F1):", metrics.sort_values("f1_macro").iloc[-1]["model"])