import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

LABELS = ["Positive", "Negative", "Neutral", "Mixed"]

# 1. Charger les données
df = pd.read_csv("data/labeled_comments.csv")
df = df.dropna(subset=["comment_text", "sentiment"])
df = df[df["sentiment"].isin(LABELS)]   # exclut aussi "Unclear"

X = df["comment_text"].astype(str)
y = df["sentiment"]

# 2. Vérifier le dataset
counts = y.value_counts()
print("Total comments:", len(df))
print(counts, "\n")

n_classes = y.nunique()
if n_classes < 2:
    raise ValueError("Il faut au moins 2 classes de sentiment dans le CSV.")

# 3. Choisir la taille du test set
#    Avec stratify, le test ET le train doivent contenir
#    au moins une ligne par classe.
n_test = max(int(round(len(df) * 0.20)), n_classes)
can_stratify = (counts.min() >= 2) and (len(df) - n_test >= n_classes)

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42, stratify=y
    )
else:
    print("ATTENTION : dataset trop petit pour un split stratifié.")
    print("Split aléatoire utilisé, les métriques n'ont AUCUNE valeur scientifique.\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

# 4. Modèles
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)),
        ("classifier", LinearSVC(class_weight="balanced")),
    ]),
}

# 5. Entraîner, évaluer, sauvegarder
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"===== {name} =====")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, f"models/{filename}")

print("Training completed.")
