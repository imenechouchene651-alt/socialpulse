import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/final_results.csv")
df = df[df["assistant"].isin(["ChatGPT", "Gemini", "Claude"])]
os.makedirs("reports", exist_ok=True)

counts = pd.crosstab(df["assistant"], df["sentiment"])
pct = (pd.crosstab(df["assistant"], df["sentiment"], normalize="index") * 100).round(2)
topics = pd.crosstab(df["assistant"], df["topic_name"])

counts.to_csv("reports/sentiment_counts.csv")
pct.to_csv("reports/sentiment_percentages.csv")
topics.to_csv("reports/topics_by_assistant.csv")

print("Commentaires par assistant:\n", df["assistant"].value_counts(), "\n")
print("Sentiments (%):\n", pct, "\n")
print("Topics par assistant:\n", topics)

pct.plot(kind="bar", figsize=(8, 5))
plt.ylabel("% of comments")
plt.title("Sentiment by assistant")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/sentiment_by_assistant.png", dpi=150)