import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

N_TOPICS = 5
TOPIC_NAMES = {0: "Topic 0", 1: "Topic 1", 2: "Topic 2", 3: "Topic 3", 4: "Topic 4"}

path = "data/processed/cleaned_comments.csv"
if not os.path.exists(path):
    path = "data/labeled_comments.csv"
print("Lecture de:", path)

df = pd.read_csv(path).dropna(subset=["comment_text"]).reset_index(drop=True)

vectorizer = CountVectorizer(
    stop_words="english",
    max_df=1.0 if len(df) < 50 else 0.90,
    min_df=2 if len(df) >= 50 else 1,
    max_features=5000,
    ngram_range=(1, 2),
)
X = vectorizer.fit_transform(df["comment_text"].astype(str))
if X.shape[1] == 0:
    raise ValueError("Aucun terme exploitable. Vérifie les données.")

lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42,
                                max_iter=20, learning_method="batch")
doc_topics = lda.fit_transform(X)

df["topic_id"] = np.argmax(doc_topics, axis=1)
df["topic_name"] = df["topic_id"].map(TOPIC_NAMES)

words = vectorizer.get_feature_names_out()
rows = []
for i, comp in enumerate(lda.components_):
    top = [words[j] for j in comp.argsort()[-10:][::-1]]
    rows.append({"topic_id": i, "keywords": ", ".join(top)})
    print(f"Topic {i}: {', '.join(top)}")

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

pd.DataFrame(rows).to_csv("reports/topic_keywords.csv", index=False)
df.to_csv("data/processed/comments_with_topics.csv", index=False)
joblib.dump(vectorizer, "models/lda_vectorizer.joblib")
joblib.dump(lda, "models/lda_model.joblib")
print("Terminé.")