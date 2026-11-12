import os
import pandas as pd

def read(p):
    return pd.read_csv(p) if os.path.exists(p) else None

models = read("reports/model_comparison.csv")
pct = read("reports/sentiment_percentages.csv")
counts = read("reports/sentiment_counts.csv")
kw = read("reports/topic_keywords.csv")
kappa = read("reports/kappa.csv")
labeled = read("data/labeled_comments.csv")
final = read("data/processed/final_results.csv")

L = ["# SocialPulse AI : Rapport Student 2 (AI/ML)\n"]

L.append("## 1. Données")
if labeled is not None:
    L.append(f"- Commentaires annotés : {len(labeled)}")
    L.append("- Répartition des labels :\n\n" + labeled["sentiment"].value_counts().to_markdown() + "\n")
if final is not None:
    L.append(f"- Commentaires analysés (prédictions) : {len(final)}")
if kappa is not None:
    L.append(f"- Cohen's Kappa : {kappa.iloc[0]['kappa']} sur {kappa.iloc[0]['n']} commentaires")

L.append("\n## 2. Performance des modèles")
if models is not None:
    L.append(models.to_markdown(index=False))
    best = models.sort_values("f1_macro").iloc[-1]
    L.append(f"\nMeilleur modèle selon le Macro F1 : **{best['model']}** ({best['f1_macro']}).")
    L.append("\nMatrices de confusion : `reports/confusion_matrix_*.png`")

L.append("\n## 3. Topics découverts (LDA)")
if kw is not None:
    L.append(kw.to_markdown(index=False))
    L.append("\n*Nom donné à chaque topic après lecture de commentaires : à compléter.*")

L.append("\n## 4. Comparaison des assistants")
if counts is not None:
    L.append("Nombre de commentaires :\n\n" + counts.to_markdown(index=False))
if pct is not None:
    L.append("\nPourcentages :\n\n" + pct.to_markdown(index=False))
    for s in ["Positive", "Negative"]:
        if s in pct.columns:
            r = pct.loc[pct[s].idxmax()]
            L.append(f"\n- Plus forte proportion de commentaires {s} : **{r.iloc[0]}** ({r[s]} %)")

L.append("\n## 5. Interprétation (à rédiger)")
L.append("- Que apprécient les utilisateurs de chaque assistant ?\n- Quelles critiques reviennent ?\n- Quels sujets dominent par assistant ?")

L.append("\n## 6. Limites")
L.append("- Échantillon limité, non représentatif de toute l'opinion publique.\n"
         "- Les sentiments du dashboard sont **prédits par le modèle**, pas annotés manuellement.\n"
         "- Sarcasme et commentaires ambigus difficiles à classer.\n"
         "- Topics nommés manuellement (part de subjectivité).")

os.makedirs("reports", exist_ok=True)
open("reports/final_report.md", "w", encoding="utf-8").write("\n".join(L))
print("Rapport généré : reports/final_report.md")
