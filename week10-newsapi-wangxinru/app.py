import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Week 10 Open API — News Recommendation System • wang xinru",
                   page_icon="📰", layout="wide")

st.title("Week 10 Open API — News Recommendation System • wang xinru")
st.caption("Enter your NewsAPI key on the left → click 'Start' to get recommendations.")

# Sidebar
with st.sidebar:
    st.header("🔑 API Credentials")
    hidden = st.checkbox("Hide API Key", value=True)
    api_key = st.text_input("NewsAPI Key", type="password" if hidden else "default", placeholder="YOUR_NEWSAPI_KEY")
    st.divider()

    st.header("🔍 Query Settings")
    query = st.text_input("Keyword (optional)", placeholder="e.g., ai, economy, sports")
    country = st.selectbox("Country", ["", "us", "gb", "kr", "jp", "cn", "de", "fr", "in"])
    category = st.selectbox("Category", ["", "business", "entertainment", "general", "health", "science", "sports", "technology"])
    sort_by = st.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"])
    page_size = st.slider("Results to fetch", 10, 50, 25, step=5)
    st.divider()
    run = st.button("▶️ Start", use_container_width=True)

# CSS dark theme
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1116; }
[data-testid="stSidebar"] { background: #0b0d12; }
h1,h2,h3,h4,h5,p,span,div,label { color: #e5e7eb !important; }
.glass { background: rgba(255,255,255,0.06); border-radius:16px; padding:14px; margin:8px 0; }
a { color: #8ab4ff !important; }
</style>
""", unsafe_allow_html=True)

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

def score_article(row, keyword: str = ""):
    score = 0.0
    if pd.notnull(row["publishedAt"]):
        pub_time = row["publishedAt"]
        try:
            if hasattr(pub_time, "tzinfo") and pub_time.tzinfo is not None:
                pub_time = pub_time.tz_convert(None)
            hours = (pd.Timestamp.utcnow() - pub_time).total_seconds() / 3600
        except Exception:
            hours = 9999
        if hours < 6: score += 5
        elif hours < 24: score += 4
        elif hours < 72: score += 3
        elif hours < 168: score += 2
        else: score += 1

    title = row.get("title") or ""
    if 40 <= len(title) <= 100:
        score += 1.5

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
        df["score_norm"] = ((df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min()) * 100).round(1)
    else:
        df["score_norm"] = 0
    return df

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
            st.subheader("🏆 Top Recommendations")
            for _, row in df.sort_values("score_norm", ascending=False).head(10).iterrows():
                st.markdown(f"""
                <div class='glass'>
                    <h4>{row['title'] or 'Untitled'}</h4>
                    <p>{row['description'] or ''}</p>
                    <p style='opacity:.7'>{row['source'] or 'Unknown'} • {row['publishedAt'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['publishedAt']) else ''}</p>
                    <b>Score:</b> {row['score_norm']} / 100<br>
                    <a href="{row['url']}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
            st.subheader("📊 Insights")
            src_counts = df["source"].value_counts().reset_index()
            src_counts.columns = ["Source", "Count"]
            fig1 = px.bar(src_counts, x="Source", y="Count", title="Articles by Source")
            st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Set your query in the sidebar and click **Start** to see results.")
