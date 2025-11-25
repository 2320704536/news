import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Emotion News + Ice Crystal", layout="wide")

st.title("📰 News + ❄️ Emotion Ice Crystal Demo")
st.caption("This is a minimal working demo. If this works, we integrate it into your full app.")

# =========================================
#  Setup
# =========================================

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

COLOR_TABLE = {
    "joy": "#E8C066",
    "positive": "#78AEEB",
    "neutral": "#9AA0A6",
    "sadness": "#4FB3C8",
    "anger": "#FF4D4D",
    "negative": "#D85C5C",
}

# =========================================
#  Sidebar
# =========================================

with st.sidebar:
    api_key = st.text_input("NewsAPI Key", type="password")
    query = st.text_input("Keyword", "ai")
    run = st.button("▶ Fetch")

# =========================================
#  Emotion Functions
# =========================================

def detect_emotion(t):
    s = sia.polarity_scores(t)
    c = s["compound"]
    if c >= 0.4:
        return "joy"
    elif c >= 0.05:
        return "positive"
    elif c <= -0.4:
        return "anger"
    elif c <= -0.1:
        return "sadness"
    else:
        return "neutral"

# =========================================
#  Visualization
# =========================================

def draw_crystal(df):
    np.random.seed(0)
    df["x"] = np.random.randn(len(df)).cumsum()
    df["y"] = np.random.randn(len(df)).cumsum()
    df["size"] = 30

    fig = go.Figure(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=df["size"],
            color=df["color"],
            line=dict(color="white", width=1),
            opacity=0.95
        ),
        text=df["title"],
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

    fig.update_layout(
        title="❄ Emotion Ice Crystal Map",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font_color="white"
    )
    return fig

# =========================================
#  Main
# =========================================

if run:
    if not api_key:
        st.error("Please enter API key.")
    else:
        url = "https://newsapi.org/v2/everything"
        r = requests.get(url, params={"q": query, "apiKey": api_key, "pageSize": 20})
        data = r.json().get("articles", [])

        if not data:
            st.warning("No results.")
        else:
            df = pd.DataFrame(data)
            df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
            df["emotion"] = df["text"].apply(detect_emotion)
            df["color"] = df["emotion"].map(COLOR_TABLE)

            st.subheader("📝 Articles + Emotion")
            st.dataframe(df[["title", "emotion"]])

            st.subheader("❄ Emotion Ice Crystal")
            fig = draw_crystal(df)
            st.plotly_chart(fig, use_container_width=True)
