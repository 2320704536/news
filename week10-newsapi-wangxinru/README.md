# 📰 Week 10 Open API — News Recommendation System • wang xinru

A dark-themed Streamlit app that recommends news using **NewsAPI**.

## Features
- Sidebar API key input (show/hide)
- Search by keyword or browse top headlines by country/category
- Recommendation score (recency, keyword match, title length)
- Top recommendations with scores
- Analytics: articles by source, timeline over time
- Fully deployable to Streamlit Community Cloud

## Setup
1. Get a free API key at https://newsapi.org/
2. Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push to GitHub → Deploy on Streamlit Cloud → set `app.py` as the main file.
