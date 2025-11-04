import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone
import plotly.express as px

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="Week 10 Open API — News Recommendation System • wang xinru",
                   page_icon="📰", layout="wide")

TITLE = "Week 10 Open API — News Recommendation System • wang xinru"
st.markdown(f"<h1 style='margin-bottom:0'>{TITLE}</h1>", unsafe_allow_html=True)
st.caption("Enter your NewsAPI key on the left → click 'Start' to get recommendations.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔑 API Credentials")
    hidden = st.checkbox("Hide API Key", value=True)
    api_key = st.text_input("NewsAPI Key", type="password" if hidden else "default", placeholder="YOUR_NEWSAPI_KEY")
    st.divider()

    st.header("🔍 Query Settings")
    query = st.text_input("Keyword (optional)", placeholder="e.g., ai, economy, sports")
    country = st.selectbox("Country (for Top Headlines)", ["", "us", "gb", "kr", "jp", "cn", "de", "fr", "in"])
    category = st.selectbox("Category (optional)", ["", "business", "entertainment", "general", "health", "science", "sports", "technology"])
    sort_by = st.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"])
    page_size = st.slider("Results to fetch", 10, 50, 25, step=5)
    st.divider()
    run = st.button("▶️ Start", use_container_width=True)

# -------------------- STYLES (dark tech) --------------------
st.markdown("""
<style>
/* Dark tech look */
:root {
  --card-bg: rgba(255,255,255,0.06);
}
[data-testid="stAppViewContainer"] {
  background: #0f1116;
}
[data-testid="stSidebar"] {
  background: #0b0d12;
  border-right: 1px solid rgba(255,255,255,0.06);
}
h1,h2,h3,h4,h5,h6, p, span, label, div, code {
  color: #e5e7eb !important;
}
/* Glass cards */
.glass {
  background: var(--card-bg);
  border-radius: 16px;
  padding: 16px 18px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}
/* Accent */
a { color: #8ab4ff !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
NEWSAPI_BASE = "https://newsapi.org/v2"

@st.cache_data(ttl=300)
def fetch_news(api_key: str, query: str, country: str, category: str, sort_by: str, page_size: int):
    if not api_key:
        return {"articles": [], "error": "Missing API key."}

    headers = {"User-Agent": "streamlit-news-app"}
    params_common = {"apiKey": api_key, "pageSize": page_size}

    try:
        if query:
            # Everything endpoint (last 30 days; supports sortBy)
            url = f"{NEWSAPI_BASE}/everything"
            params = {**params_common, "q": query, "sortBy": sort_by, "language": "en"}
        else:
            # Top headlines by country/category
            url = f"{NEWSAPI_BASE}/top-headlines"
            params = {**params_common}
            if country:
                params["country"] = country
            if category:
                params["category"] = category
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            try:
                msg = r.json().get("message", r.text)
            except Exception:
                msg = r.text
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
            "publishedAt": a.get("publishedAt"),
            "content": a.get("content")
        })
    df = pd.DataFrame(rows)
    # parse dates
    if "publishedAt" in df:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df

def score_article(row, keyword:str=""):
    score = 0.0
    # recency: newer is better
    if pd.notnull(row["publishedAt"]):
        hours = (pd.Timestamp.utcnow() - row["publishedAt"].tz_convert(None)).total_seconds()/3600 if getattr(row["publishedAt"], "tzinfo", None) else (pd.Timestamp.utcnow() - row["publishedAt"]).total_seconds()/3600
        if hours < 6: score += 5
        elif hours < 24: score += 4
        elif hours < 72: score += 3
        elif hours < 168: score += 2
        else: score += 1
    # title length: medium length preferred
    title = row.get("title") or ""
    if 40 <= len(title) <= 100: score += 1.5
    # keyword presence
    text = (str(row.get("title") or "") + " " + str(row.get("description") or "")).lower()
    for kw in [k.strip() for k in (keyword or "").split(",") if k.strip()]:
        if kw.lower() in text:
            score += 2.0
    return score

def add_scores(df, keyword):
    if df.empty:
        return df
    df = df.copy()
    df["score"] = df.apply(lambda r: score_article(r, keyword), axis=1)
    # normalize within 0-100
    if df["score"].max() > 0:
        df["score_norm"] = ((df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min()) * 100).round(1)
    else:
        df["score_norm"] = 0
    return df

# -------------------- MAIN --------------------
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
            # Header metrics
            colA, colB, colC = st.columns(3)
            colA.markdown(f"<div class='glass'><h3 style='margin:0'>{len(df)}</h3><p style='margin:0'>Articles</p></div>", unsafe_allow_html=True)
            colB.markdown(f"<div class='glass'><h3 style='margin:0'>{df['source'].nunique()}</h3><p style='margin:0'>Sources</p></div>", unsafe_allow_html=True)
            colC.markdown(f"<div class='glass'><h3 style='margin:0'>{df['score_norm'].mean():.1f}</h3><p style='margin:0'>Avg. Recommendation</p></div>", unsafe_allow_html=True)

            st.subheader("🏆 Top Recommendations")
            topk = st.slider("Show top N", 5, min(20, len(df)), min(10, len(df)), step=1)
            show_cols = st.columns(2)
            for i, (_, row) in enumerate(df.sort_values("score_norm", ascending=False).head(topk).iterrows()):
                with show_cols[i % 2]:
                    st.markdown(f"""
                    <div class='glass' style='margin-bottom:12px'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <h4 style='margin:0'>{row['title'] or 'Untitled'}</h4>
                            <span style='font-weight:bold'>{row['score_norm']} / 100</span>
                        </div>
                        <p style='opacity:.85;margin:.25rem 0'>{row['description'] or ''}</p>
                        <p style='opacity:.6;margin:.25rem 0'>{row['source'] or 'Unknown'} • {row['publishedAt'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['publishedAt']) else ''}</p>
                        <a href="{row['url']}" target="_blank">Read more →</a>
                    </div>
                    """, unsafe_allow_html=True)

            st.subheader("📊 Insights")
            # Source distribution
            src_counts = df["source"].value_counts().reset_index()
            src_counts.columns = ["Source", "Count"]
            fig1 = px.bar(src_counts, x="Source", y="Count", title="Articles by Source")
            st.plotly_chart(fig1, use_container_width=True)

            # Timeline
            time_df = df.dropna(subset=["publishedAt"]).copy()
            time_df["datehour"] = time_df["publishedAt"].dt.floor("H")
            tl = time_df.groupby("datehour").size().reset_index(name="Articles")
            fig2 = px.line(tl, x="datehour", y="Articles", markers=True, title="Articles over Time")
            st.plotly_chart(fig2, use_container_width=True)

            # Raw table
            with st.expander("Raw data table"):
                st.dataframe(df[["source","title","publishedAt","score_norm","url"]], use_container_width=True, hide_index=True)

else:
    st.info("Set your query in the sidebar and click **Start** to see results.")
