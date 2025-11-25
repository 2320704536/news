# ============================================================
#   News Recommendation + Emotion CrystalMix • wang xinru
#   Full integrated version (for Streamlit Cloud or local)
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# ---------------------- Streamlit Setup ---------------------
st.set_page_config(
    page_title="News + Emotion CrystalMix • wang xinru",
    page_icon="❄️",
    layout="wide"
)

st.title("📰 News + ❄ Emotion CrystalMix • wang xinru")
st.caption("Using VADER + Emotion→Color + CrystalMix visual system.")

# ---------------------- NLTK Init ---------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ============================================================
#          Emotion RGB Palette (20+ emotions)
# ============================================================

EMOTION_RGB = {
    "joy": (255, 200, 60),
    "love": (255, 95, 150),
    "pride": (255, 160, 90),
    "hope": (210, 220, 255),
    "calm": (120, 170, 255),
    "trust": (110, 180, 200),
    "curiosity": (150, 200, 255),
    "awe": (175, 150, 255),
    "nostalgia": (255, 180, 140),
    "surprise": (255, 240, 140),
    "mixed": (160, 120, 200),
    "anger": (245, 60, 60),
    "fear": (160, 90, 255),
    "sadness": (60, 150, 200),
    "anxiety": (100, 130, 180),
    "disgust": (120, 160, 90),
    "boredom": (180, 180, 180),
    "neutral": (160, 160, 160),
}

# ============================================================
#                Expanded Emotion Classifier
# ============================================================

def classify_emotion_expanded(row):
    neg = row["neg"]
    neu = row["neu"]
    pos = row["pos"]
    compound = row["compound"]

    # Strong positive
    if compound > 0.7 and pos > 0.6:
        return "joy"
    if compound > 0.6 and pos > 0.5:
        return "love"
    if compound > 0.5 and pos > 0.4:
        return "pride"
    if compound > 0.4 and pos > 0.4:
        return "hope"

    # Positive nuanced
    if compound > 0.2 and neu > 0.4:
        return random.choice(["calm", "trust", "curiosity", "awe", "nostalgia"])

    if pos > 0.3 and neu > 0.4:
        return "surprise"

    # Mixed
    if pos > 0.25 and neg > 0.25:
        return "mixed"

    # Negative
    if compound < -0.7 and neg > 0.5:
        return "anger"
    if compound < -0.6:
        return "fear"
    if compound < -0.5:
        return "sadness"
    if compound < -0.3 and neg > 0.3:
        return "anxiety"
    if compound < -0.2:
        return "disgust"

    # Neutral / boredom
    if abs(compound) < 0.05:
        return "boredom" if neu > 0.8 else "neutral"

    return "neutral"

# ============================================================
#                  Crystal Shape Generator
# ============================================================

def crystal_shape(cx, cy, r, wobble=0.25, sides_min=5, sides_max=10):
    n = random.randint(sides_min, sides_max)
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    random.shuffle(angles)

    pts = []
    for a in angles:
        rr = r * (1 + wobble * (random.random() - 0.5))
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((x, y))
    pts.append(pts[0])
    return pts

# ============================================================
#            Soft Polygon Drawing with Glow
# ============================================================

def draw_polygon_soft(base, points, color, alpha=150, blur_px=6, edge_width=1):
    layer = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(layer)

    draw.polygon(points, fill=(*color, alpha))
    if edge_width > 0:
        draw.line(points, fill=(255,255,255,200), width=edge_width)

    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))
    base.alpha_composite(layer)
    return base

# ============================================================
#                    CrystalMix Renderer
# ============================================================

def render_crystalmix(df, width=1400, height=900, seed=123,
                      shapes_per_emotion=5,
                      min_size=40, max_size=140,
                      wobble=0.25,
                      blur_px=6,
                      layers=4,
                      bg_color=(15,17,22)):
    
    random.seed(seed)
    np.random.seed(seed)

    base = Image.new("RGBA", (width, height), bg_color + (255,))

    emotions = df["emotion"].value_counts().index.tolist()
    if len(emotions) == 0:
        emotions = ["joy", "love", "curiosity"]

    margin = 80

    for _layer in range(layers):
        for emo in emotions:
            color = EMOTION_RGB.get(emo, (170,170,170))
            for _ in range(shapes_per_emotion):

                cx = random.randint(margin, width-margin)
                cy = random.randint(margin, height-margin)
                r = random.randint(min_size, max_size)

                pts = crystal_shape(cx, cy, r, wobble=wobble)
                base = draw_polygon_soft(base, pts, color,
                                         alpha=150,
                                         blur_px=blur_px,
                                         edge_width=1)

    return base.convert("RGB")

# ============================================================
#                      Sidebar Controls
# ============================================================

with st.sidebar:
    st.header("🔧 Controls")

    api_key = st.secrets["NEWS_API_KEY"]  # use secrets
    query = st.text_input("Search keyword", "AI")
    page_size = st.slider("Results", 10, 50, 20)

    # Crystal parameters
    seed = st.slider("Crystal Seed", 1, 999, 123)
    layers = st.slider("Layers", 1, 8, 4)
    wobble = st.slider("Wobble (shape randomness)", 0.0, 1.0, 0.25)
    blur_px = st.slider("Blur (soft glow)", 0, 15, 6)

    run = st.button("▶ Run")

# ============================================================
#                     Fetch NewsAPI
# ============================================================

def fetch_news(api_key, query, page_size):
    url = "https://newsapi.org/v2/everything"
    r = requests.get(url, params={
        "q": query,
        "apiKey": api_key,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt"
    })

    if r.status_code != 200:
        st.error("API Error: " + r.json().get("message", ""))
        return pd.DataFrame()

    data = r.json().get("articles", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["source"] = df["source"].apply(lambda s: s.get("name") if isinstance(s, dict) else s)
    return df[["timestamp", "text", "source", "title", "description", "url"]]

# ============================================================
#                          Main
# ============================================================

if run:

    df = fetch_news(api_key, query, page_size)

    if df.empty:
        st.warning("No results.")
        st.stop()

    # ---------- VADER scores ----------
    vader_res = df["text"].apply(sia.polarity_scores)
    df["neg"] = vader_res.apply(lambda x: x["neg"])
    df["neu"] = vader_res.apply(lambda x: x["neu"])
    df["pos"] = vader_res.apply(lambda x: x["pos"])
    df["compound"] = vader_res.apply(lambda x: x["compound"])

    # ---------- Expanded Emotion ----------
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

    # ============================================================
    #                 Glass Card Article Display
    # ============================================================

    st.subheader("🧊 News Cards with Emotion")

    for _, row in df.iterrows():
        st.markdown(f"""
        <div class='glass'>
            <h4>{row['title']}</h4>
            <p>{row['description']}</p>
            <p style='opacity:.7'>{row['source']} — {row['timestamp']}</p>
            <b>Emotion:</b> {row['emotion']}<br>
            <a href="{row['url']}" target="_blank">Read more →</a>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    #                     Source Distribution
    # ============================================================

    st.subheader("📊 Source Distribution")

    src = df["source"].value_counts().reset_index()
    src.columns = ["Source", "Count"]
    fig = px.bar(src, x="Source", y="Count", title="Articles by Source")
    fig.update_layout(paper_bgcolor="#0f1116", plot_bgcolor="#0f1116", font_color="#fff")
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    #                   Emotion CrystalMix Render
    # ============================================================

    st.subheader("❄ Emotion CrystalMix — Full Ice Crystal Rendering")

    crystal = render_crystalmix(
        df,
        width=1400,
        height=900,
        seed=seed,
        shapes_per_emotion=5,
        layers=layers,
        blur_px=blur_px,
        wobble=wobble
    )

    st.image(crystal, use_container_width=True)
    st.success("CrystalMix rendered successfully!")
