import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ========== INITIAL SETUP ==========
st.set_page_config(
    page_title="News Recommendation + Emotion Crystal • wang xinru",
    page_icon="❄️",
    layout="wide"
)

st.title("📰 News Recommendation + ❄️ Emotion Crystal Visualization • wang xinru")
st.caption("Enter your NewsAPI key on the left → click 'Start' to get recommendations.")

# NLTK setup
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ---------- Emotion Color Maps ----------
EMOTION_MAP = {
    "positive": "calm_blue",
    "negative": "danger_red",
    "neutral": "fog_gray",
    "joy": "warm_gold",
    "anger": "neon_red",
    "sadness": "cold_teal",
    "fear": "purple_shadow",
    "surprise": "mint_green"
}

COLOR_TABLE = {
    "calm_blue": "#78AEEB",
    "danger_red": "#D85C5C",
    "fog_gray": "#9AA0A6",
    "warm_gold": "#E8C066",
    "neon_red": "#FF4D4D",
    "cold_teal": "#4FB3C8",
    "purple_shadow": "#8C6FF7",
    "mint_green": "#6EE7B7"
}

# ---------- CSS ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1116; }
[data-testid="stSidebar"] { background: #0b0d12; }
h1,h2,h3,h4,h5,p,span,div,label { color: #e5e7eb !important; }
.glass { background: rgba(255,255,255,0.06); border-radius:16px; padding:14px; margin:8px 0; }
a { color: #8ab4ff !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔑 API Credentials")
    hidden = st.checkbox("Hide API Key", value=True)
    api_key = st.text_input("NewsAPI Key",
                            type="password" if hidden else "default",
                            placeholder="YOUR_NEWSAPI_KEY")

    st.divider()
    st.header("🔍 Query Settings")
    query = st.text_input("Keyword (optional)", placeholder="e.g., ai, economy")
    country = st.selectbox("Country", ["", "us", "gb", "kr", "jp", "cn", "de", "fr", "in"])
    category = st.selectbox("Category", ["", "business", "entertainment", "general",
                                         "health", "science", "sports", "technology"])
    sort_by = st.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"])
    page_size = st.slider("Results to fetch", 10, 50, 25, step=5)

    st.divider()
    st.header("🎬 Film Color Shift")
    color_shift = st.slider("Hue Rotation (visual FX)", -180, 180, 0)

    run = st.button("▶️ Start", use_container_width=True)


# =============================================================
# ---------------------- CORE FUNCTIONS -----------------------
# =============================================================

NEWSAPI_BASE = "https://newsapi.org/v2"

@st.cache_data(ttl=300)
def fetch_news(api_key, query, country, category, sort_by, page_size):
    if not api_key:
        return {"articles": [], "error": "Missing API key."}

    headers = {"User-Agent": "streamlit-news-app"}
    params_common = {"apiKey": api_key, "pageSize": page_size}

    try:
        if query:
            url = f"{NEWSAPI_BASE}/everything"
            params = {**params_common, "q": query, "sortBy": sort_by, "language": "en"}
        else:
            url = f"{NEWSAPI_BASE}/top-headlines"
            params = {**params_common}
            if country:
                params["country"] = country
            if category:
                params["category"] = category

        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            msg = r.json().get("message", r.text)
            return {"articles": [], "error": f"API error: {msg}"}

        data = r.json()
        return {"articles": data.get("articles", []), "error": None}

    except Exception as e:
        return {"articles": [], "error": str(e)}


def to_dataframe(articles):
    if not articles:
        return pd.DataFrame(columns=["source","author","title","description","url","publishedAt"])
    rows = []
    for a in articles:
        rows.append({
            "source": (a.get("source") or {}).get("name"),
            "author": a.get("author"),
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt")
        })
    df = pd.DataFrame(rows)
    if "publishedAt" in df:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df


# -------------- Recommendation Score --------------
def score_article(row, keyword: str = ""):
    score = 0.0

    # Recency
    if pd.notnull(row["publishedAt"]):
        pub_time = row["publishedAt"]
        try:
            if hasattr(pub_time, "tzinfo") and pub_time.tzinfo is not None:
                pub_time = pub_time.tz_convert(None)
            hours = (pd.Timestamp.utcnow() - pub_time).total_seconds() / 3600
        except:
            hours = 9999

        if hours < 6: score += 5
        elif hours < 24: score += 4
        elif hours < 72: score += 3
        elif hours < 168: score += 2
        else: score += 1

    # Title length
    title = row.get("title") or ""
    if 40 <= len(title) <= 100:
        score += 1.5

    # Keyword match
    text = (str(row.get("title") or "") + " " + str(row.get("description") or "")).lower()
    for kw in [k.strip() for k in (keyword or "").split(",") if k.strip()]:
        if kw.lower() in text:
            score += 2.0

    return score


def add_scores(df, keyword):
    if df.empty: return df
    df = df.copy()
    df["score"] = df.apply(lambda r: score_article(r, keyword), axis=1)

    if df["score"].max() > 0:
        df["score_norm"] = ((df["score"] - df["score"].min()) /
                            (df["score"].max() - df["score"].min()) * 100).round(1)
    else:
        df["score_norm"] = 0

    return df


# -------------- Emotion Detection --------------
def detect_emotion(text):
    """Return primary + secondary emotions."""
    if not text:
        return ["neutral"]

    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # Primary sentiment
    if compound >= 0.4:
        primary = "joy"
    elif compound >= 0.05:
        primary = "positive"
    elif compound <= -0.4:
        primary = "anger"
    elif compound <= -0.1:
        primary = "sadness"
    else:
        primary = "neutral"

    # Secondary emotions
    secondary = []
    if scores["neg"] > 0.3: secondary.append("negative")
    if scores["neu"] > 0.6: secondary.append("neutral")
    if scores["pos"] > 0.4: secondary.append("joy")
    if not secondary:
        secondary = ["neutral"]

    return list(set([primary] + secondary))


def add_emotion(df):
    df = df.copy()
    df["emotion"] = df.apply(lambda r:
        detect_emotion((str(r["title"]) + " " + str(r["description"])).strip()),
        axis=1
    )
    df["emotion_primary"] = df["emotion"].apply(lambda x: x[0])
    df["color"] = df["emotion_primary"].apply(
        lambda emo: COLOR_TABLE.get(EMOTION_MAP.get(emo, "fog_gray"))
    )
    return df


# -------------- Ice Crystal Scatter --------------
def generate_ice_crystal(df, hue_shift=0):
    """Voronoi-style ice crystal scatter."""
    df = df.copy()
    n = len(df)
    np.random.seed(42)

    # Random crystal coordinates
    df["x"] = np.random.randn(n).cumsum()
    df["y"] = np.random.randn(n).cumsum()
    df["size"] = df["score_norm"].fillna(10)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=df["size"],
            color=df["color"],
            opacity=0.92,
            line=dict(color="#FFFFFF", width=1)
        ),
        text=df["title"],
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

    fig.update_layout(
        title="❄️ Emotion Ice Crystal Map (Film Graded)",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    return fig


# =============================================================
# ---------------------- MAIN LOGIC ---------------------------
# =============================================================
if not api_key:
    st.info("👉 Please enter your **NewsAPI Key** in the left sidebar, then click **Start**.")

elif run:
    with st.spinner("Fetching news..."):
        result = fetch_news(api_key, query, country, category, sort_by, page_size)

    if result["error"]:
        st.error(result["error"])

    else:
        df = to_dataframe(result["articles"])

        if df.empty:
            st.warning("No articles found. Try another keyword/country/category.")

        else:
            df = add_scores(df, query or "")
            df = add_emotion(df)

            # ---------- Top Recommendations ----------
            st.subheader("🏆 Top Recommendations")
            for _, row in df.sort_values("score_norm", ascending=False).head(10).iterrows():
                st.markdown(f"""
                <div class='glass'>
                    <h4>{row['title'] or 'Untitled'}</h4>
                    <p>{row['description'] or ''}</p>
                    <p style='opacity:.7'>{row['source'] or 'Unknown'} • 
                    {row['publishedAt'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['publishedAt']) else ''}</p>
                    <b>Score:</b> {row['score_norm']} / 100<br>
                    <b>Emotion:</b> {row['emotion']}<br>
                    <a href="{row['url']}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)

            # ---------- Insights ----------
            st.subheader("📊 Source Distribution")
            src_counts = df["source"].value_counts().reset_index()
            src_counts.columns = ["Source", "Count"]
            fig1 = px.bar(src_counts, x="Source", y="Count", title="Articles by Source")
            fig1.update_layout(paper_bgcolor="#0f1116", plot_bgcolor="#0f1116", font_color="#e5e7eb")
            st.plotly_chart(fig1, use_container_width=True)

            # ---------- Emotion Crystal ----------
            st.subheader("❄️ Emotion Ice Crystal Map")
            fig2 = generate_ice_crystal(df, hue_shift=color_shift)
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Set your query in the sidebar and click **Start** to see results.")
