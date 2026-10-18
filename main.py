# import nltk
# nltk.download('vader_lexicon')

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "articles.db"

API_KEY = (os.getenv("API_KEY") or "").strip()
API_URL = "https://newsapi.org/v2/top-headlines"


@st.cache_data
def fetch_articles():
    if not API_KEY:
        raise ValueError("Set API_KEY in your .env file to your NewsAPI key.")

    try:
        r = requests.get(
            API_URL,
            headers={"X-Api-Key": API_KEY},
            params={"category": "business"},
            timeout=30,
        )
        r.raise_for_status()

    except requests.Timeout:
        raise ValueError("The news request timed out. Please try again.") from None

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ValueError(
            f"NewsAPI returned HTTP {status}. Check your API key, plan and request quota."
        ) from None

    except requests.RequestException:
        raise ValueError(
            "Could not connect to NewsAPI. Check your connection and try again."
        ) from None

    try:
        j = r.json()

    except ValueError:
        raise ValueError(
            "NewsAPI returned invalid JSON. Please try again later."
        ) from None

    if not isinstance(j, dict) or j.get("status") != "ok":
        raise ValueError(
            "NewsAPI could not complete the request. Check your API key and quota."
        )

    if not isinstance(j.get("articles"), list):
        raise ValueError("NewsAPI returned an unexpected articles format.")

    return j["articles"]


def calculate_sentiment(articles):
    data = []
    sentiments = []
    compounds = []

    for a in articles:
        if not isinstance(a, dict):
            continue

        title, url = a.get("title"), a.get("url")

        if not isinstance(title, str) or not title.strip() or title == "[Removed]":
            continue

        if not isinstance(url, str) or not url.strip():
            continue

        source = a.get("source")
        publisher = source.get("name") if isinstance(source, dict) else None
        publisher = publisher if isinstance(publisher, str) else "Unknown"
        pub_date = a.get("publishedAt")
        pub_date = pub_date if isinstance(pub_date, str) else ""

        description = a.get("description")
        description = description if isinstance(description, str) else ""
        content = a.get("content")
        content = content if isinstance(content, str) else ""

        data.append([publisher, pub_date, title, url])

        text = f"{title}. {description} {content}"
        comp = sia.polarity_scores(text)["compound"]
        compounds.append(comp)

        if comp >= 0.05:
            sentiments.append("pos")
        elif comp <= -0.05:
            sentiments.append("neg")
        else:
            sentiments.append("neu")

    return data, sentiments, compounds


def save_articles(data, sentiment, compound):
    if not len(data) == len(sentiment) == len(compound):
        raise ValueError("Article and sentiment counts must match.")

    rows = [
        (*article, sen, comp) for article, sen, comp in zip(data, sentiment, compound)
    ]

    with closing(sqlite3.connect(DB_PATH)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                publisher TEXT,
                pub_date TEXT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                sentiment TEXT NOT NULL,
                compound REAL NOT NULL
            )""")
        db.executemany(
            """
            INSERT INTO articles (publisher, pub_date, title, url, sentiment, compound)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO NOTHING
            """,
            rows,
        )


st.title("Financial News Sentiment Analyser")

data_load_state = st.text("Fetching data...")

try:
    sia = SentimentIntensityAnalyzer()

except LookupError:
    data_load_state.empty()
    st.error(
        "The VADER lexicon is missing. Run `python -m nltk.downloader vader_lexicon` in your project environment, "
        "then restart the app."
    )
    st.stop()

try:
    articles = fetch_articles()

except ValueError as exc:
    data_load_state.empty()
    st.error(str(exc))
    st.stop()

data_load_state.text("Fetching data...done!")

data, sentiment, compound = calculate_sentiment(articles)

if len(data) < len(articles):
    st.warning(
        f"Skipped {len(articles) - len(data)} articles with missing or invalid titles or URLs."
    )

if not data:
    st.info("No usable business news articles were returned. Please try again later.")
    st.stop()

try:
    save_articles(data, sentiment, compound)

except sqlite3.Error:
    st.warning(
        "Articles could not be saved. Check that articles.db is writable, available and has the expected schema. Analysis is still shown below."
    )

pos_score = sentiment.count("pos")
neu_score = sentiment.count("neu")
neg_score = sentiment.count("neg")

total = len(sentiment)
pos_pct = pos_score / total * 100
neu_pct = neu_score / total * 100
neg_pct = neg_score / total * 100
average = sum(compound) / total
balance = pos_pct - neg_pct

if pos_pct > 50:
    statement = "Most articles used positive language"
elif neg_pct > 50:
    statement = "Most articles used negative language"
elif neu_pct > 50:
    statement = "Most articles used neutral language"
else:
    statement = "Sentiment was mixed, with no category forming a majority"

conclusion = f"""
A total of {total} business news articles was retrieved and analysed just now. Out of that, {pos_score} had positive sentiment
({pos_pct:.1f}%), {neu_score} had neutral sentiment ({neu_pct:.1f}%), and {neg_score} had negative sentiment
({neg_pct:.1f}%). The average compound sentiment calculated was {average:+.3f} and the sentiment percentage range
calculated was {balance:+.1f}. {statement}.

These results describe the language in this news sample; they do not predict stock prices or establish investor
sentiment. The findings are limited to the retrieved articles and the text available through the API, which may be
truncated.
"""
st.write(conclusion)

st.subheader("Sentiment trends")

try:
    with closing(sqlite3.connect(DB_PATH)) as db:
        history = pd.read_sql_query(
            "SELECT pub_date, title, publisher, sentiment, compound FROM articles",
            db,
        )
except (sqlite3.Error, pd.errors.DatabaseError):
    st.warning("Could not load saved articles for the dashboard.")
else:
    history["date"] = pd.to_datetime(
        history["pub_date"], errors="coerce", utc=True
    ).dt.date
    history = history.dropna(subset=["date"])

    if history.empty:
        st.info("No dated articles are available yet.")
    else:
        date_range = st.date_input(
            "Publication date range",
            value=(history["date"].min(), history["date"].max()),
            min_value=history["date"].min(),
            max_value=history["date"].max(),
        )

        search_query = st.text_input(
            "Search headlines",
            placeholder="Enter a keyword or company name",
        ).strip()

        publishers = st.multiselect(
            "Publishers",
            options=sorted(history["publisher"].dropna().unique()),
        )

        sentiment_labels = {
            "Positive": "pos",
            "Neutral": "neu",
            "Negative": "neg",
        }

        chosen_sentiments = st.multiselect(
            "Sentiment",
            options=list(sentiment_labels),
        )

        if len(date_range) == 2:
            start, end = date_range
            selected = history[history["date"].between(start, end)]

            if search_query:
                selected = selected[
                    selected["title"].str.contains(
                        search_query,
                        case=False,
                        regex=False,
                        na=False,
                    )
                ]

            if publishers:
                selected = selected[selected["publisher"].isin(publishers)]

            if chosen_sentiments:
                codes = [sentiment_labels[label] for label in chosen_sentiments]
                selected = selected[selected["sentiment"].isin(codes)]

            if selected.empty:
                st.info("No articles match your search and filters.")
            else:
                left, right = st.columns(2)
                left.metric("Articles", len(selected))
                right.metric(
                    "Average sentiment",
                    f"{selected['compound'].mean():+.3f}",
                )

                daily_average = selected.groupby("date")["compound"].mean()
                st.caption("Average daily sentiment: −1 negative to +1 positive")
                st.line_chart(daily_average, y_label="Average sentiment")

                daily_counts = (
                    selected.groupby(["date", "sentiment"])
                    .size()
                    .unstack(fill_value=0)
                    .rename(
                        columns={
                            "pos": "Positive",
                            "neu": "Neutral",
                            "neg": "Negative",
                        }
                    )
                )
                st.caption("Daily article counts by sentiment")
                st.bar_chart(daily_counts, y_label="Articles")

                st.dataframe(
                    selected[["date", "title", "publisher", "sentiment", "compound"]],
                    hide_index=True,
                )
