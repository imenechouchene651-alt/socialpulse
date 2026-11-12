import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SocialPulse AI", page_icon="📊", layout="wide")

st.title("SocialPulse AI")
st.caption("Public Opinion Analysis: ChatGPT vs Gemini vs Claude")

# ---------- Chemins (relatifs à la racine du projet) ----------
ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    ROOT / "data" / "processed" / "final_results.csv",
    ROOT / "data" / "processed" / "comments_with_topics.csv",
    ROOT / "data" / "labeled_comments.csv",
]

DATA = next((p for p in CANDIDATES if p.exists()), None)
if DATA is None:
    st.error("Aucun fichier de données trouvé. Fichiers cherchés :")
    for p in CANDIDATES:
        st.code(str(p))
    st.info("Lance d'abord : topic_modeling.py puis predict_sentiment.py")
    st.stop()

df = pd.read_csv(DATA)
st.sidebar.title("SocialPulse AI")
st.sidebar.caption(f"Données : {DATA.name}")

# ---------- Vérification des colonnes ----------
for col in ["comment_text", "assistant", "sentiment"]:
    if col not in df.columns:
        st.error(f"Colonne manquante dans {DATA.name} : {col}")
        st.write("Colonnes trouvées :", list(df.columns))
        st.stop()

if "topic_name" not in df.columns:
    if "topic_id" in df.columns:
        df["topic_name"] = "Topic " + df["topic_id"].astype(str)
    else:
        df["topic_name"] = "N/A"

has_dates = False
if "published_date" in df.columns:
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    has_dates = df["published_date"].notna().any()

if len(df) == 0:
    st.warning("Le fichier de données est vide.")
    st.stop()

COLORS = {"Positive": "#2ca02c", "Negative": "#d62728",
          "Neutral": "#7f7f7f", "Mixed": "#ff7f0e"}

# ---------- Navigation et filtres ----------
page = st.sidebar.radio(
    "Page",
    ["Overview", "Sentiment Analysis", "Topic Analysis",
     "Trends", "Explore Comments", "Model Performance"],
)

all_assistants = sorted(df["assistant"].dropna().unique())
selected = st.sidebar.multiselect("Assistants", all_assistants, default=all_assistants)
f = df[df["assistant"].isin(selected)]

if f.empty:
    st.warning("Aucun commentaire pour ce filtre.")
    st.stop()

# ---------- Pages ----------
if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Total comments", len(f))
    c2.metric("Assistants", f["assistant"].nunique())
    c3.metric("Most discussed", f["assistant"].mode()[0])

    fig = px.pie(f, names="sentiment", title="Sentiment distribution",
                 color="sentiment", color_discrete_map=COLORS)
    st.plotly_chart(fig, width="stretch")

    vol = f["assistant"].value_counts().reset_index()
    vol.columns = ["assistant", "count"]
    st.plotly_chart(px.bar(vol, x="assistant", y="count",
                           title="Comments per assistant"),
                    use_container_width=True)

elif page == "Sentiment Analysis":
    pct = (pd.crosstab(f["assistant"], f["sentiment"], normalize="index") * 100).round(1)
    long = pct.reset_index().melt(id_vars="assistant", value_name="percent")
    fig = px.bar(long, x="assistant", y="percent", color="sentiment",
                 barmode="group", color_discrete_map=COLORS,
                 labels={"percent": "% of comments"},
                 title="Sentiment by assistant")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Percentages")
    st.dataframe(pct, use_container_width=True)
    st.subheader("Counts")
    st.dataframe(pd.crosstab(f["assistant"], f["sentiment"]), use_container_width=True)

elif page == "Topic Analysis":
    kw = ROOT / "reports" / "topic_keywords.csv"
    if kw.exists():
        st.subheader("Keywords per topic")
        st.dataframe(pd.read_csv(kw), use_container_width=True)
    else:
        st.info("reports/topic_keywords.csv introuvable. Lance topic_modeling.py.")

    t = f.groupby(["topic_name", "assistant"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(t, x="topic_name", y="count", color="assistant",
                           barmode="group", title="Topics by assistant"),
                    use_container_width=True)

elif page == "Trends":
    if not has_dates:
        st.info("Aucune date exploitable (colonne published_date absente ou vide).")
    else:
        d = f.dropna(subset=["published_date"]).copy()
        d["month"] = d["published_date"].dt.to_period("M").astype(str)
        t = d.groupby(["month", "sentiment"]).size().reset_index(name="count")
        st.plotly_chart(px.line(t, x="month", y="count", color="sentiment",
                                markers=True, color_discrete_map=COLORS,
                                title="Sentiment over time"),
                        use_container_width=True)

elif page == "Explore Comments":
    sentiments = sorted(f["sentiment"].dropna().unique())
    sel = st.multiselect("Sentiment", sentiments, default=sentiments)
    q = st.text_input("Search in comments")
    r = f[f["sentiment"].isin(sel)]
    if q:
        r = r[r["comment_text"].astype(str).str.contains(q, case=False, na=False)]
    st.write(f"{len(r)} comments")
    cols = [c for c in ["comment_text", "assistant", "sentiment", "topic_name"]
            if c in r.columns]
    st.dataframe(r[cols], use_container_width=True)

else:  # Model Performance
    mc = ROOT / "reports" / "model_comparison.csv"
    if mc.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(mc), use_container_width=True)
    else:
        st.info("reports/model_comparison.csv introuvable. Lance evaluate_model.py.")
    for name in ["logistic_regression", "linear_svm"]:
        img = ROOT / "reports" / f"confusion_matrix_{name}.png"
        if img.exists():
            st.image(str(img), caption=f"Confusion matrix - {name}", width=500)