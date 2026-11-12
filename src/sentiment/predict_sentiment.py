import joblib
import pandas as pd

df = pd.read_csv("data/processed/comments_with_topics.csv")
model = joblib.load("models/linear_svm.joblib")

df["sentiment_predicted"] = model.predict(df["comment_text"].astype(str))

if "sentiment" in df.columns:
    df = df.rename(columns={"sentiment": "sentiment_manual"})
df["sentiment"] = df["sentiment_predicted"]

df.to_csv("data/processed/final_results.csv", index=False)
print(df["sentiment"].value_counts())
print("Sauvegardé: data/processed/final_results.csv")