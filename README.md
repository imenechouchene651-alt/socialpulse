# SocialPulse AI

Analyse de l'opinion publique sur ChatGPT, Gemini et Claude : classification de sentiment, topic modeling et dashboard interactif.

## Équipe
- Student 1 : Data Engineering (collecte, nettoyage, PySpark)
- Student 2 : AI/ML et dashboard (sentiment, topics, analyse, Streamlit)

## Installation
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ordre d'exécution
```
python src/sentiment/train_model.py
python src/sentiment/evaluate_model.py
python src/topics/topic_modeling.py
python src/sentiment/predict_sentiment.py
python src/analytics/analysis_results.py
python src/analytics/generate_report.py
streamlit run dashboard/app.py
```

## Structure
- `src/sentiment/` : guide d'annotation, entraînement, évaluation, prédiction
- `src/topics/` : topic modeling (LDA)
- `src/analytics/` : analyse comparative et rapport
- `dashboard/app.py` : dashboard Streamlit
- `data/` : données annotées et traitées
- `reports/` : métriques, topics, rapport

## État actuel
Le code a été testé sur 10 commentaires fictifs. Les résultats ne sont pas des résultats de recherche. Le jeu de données réel (commentaires de Student 1 annotés) sera ajouté ensuite.

## Labels de sentiment
Positive, Negative, Neutral, Mixed (Unclear exclu de l'entraînement). Voir `src/sentiment/annotation_guide.md`.