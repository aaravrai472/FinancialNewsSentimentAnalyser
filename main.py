# import nltk
# nltk.download('vader_lexicon')

import os
import sqlite3

import requests
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
sia = SentimentIntensityAnalyzer()

db = sqlite3.connect("articles.db")
cursor = db.cursor()

API_KEY = str(os.getenv("API_KEY"))
API_URL = "https://newsapi.org/v2/top-headlines"


def calculate_sentiment(headlines):
    data = []
    sentiments = []
    compounds = []

    for a in headlines:
        publisher = a["source"]["name"]
        pub_date = a["publishedAt"]
        title = a["title"]
        url = a["url"]

        description = "" if not a["description"] else a["description"]
        content = "" if not a["content"] else a["content"]

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
    for a in enumerate(data):
        publisher, pub_date, title, url = a[1]
        sen, comp = sentiment[a[0]], compound[a[0]]

        cursor.execute(
            """
            INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO NOTHING
            """,
            (a[0], publisher, pub_date, title, url, sen, comp),
        )
        db.commit()


def search_articles(query):
    q = "%" + query + "%"
    cursor.execute("SELECT * FROM articles WHERE title LIKE (?)", (q,))
    
    results = cursor.fetchall()
    return results

def filter_articles(query, option):
    match option:
        case "date":
            cursor.execute("SELECT * FROM articles WHERE pub_date = (?)", (query,))
            return cursor.fetchall()
        
        case "company":
            q = "%" + query + "%"
            cursor.execute("SELECT * FROM articles WHERE publisher LIKE (?)", (q,))
            return cursor.fetchall()
        
        case "sentiment":
            cursor.execute("SELECT * FROM articles WHERE sentiment = (?)", (query,))
            return cursor.fetchall()
    
    return None


r = requests.get(
    API_URL, headers={"X-Api-Key": API_KEY}, params={"category": "business"}, timeout=30
)
j = r.json()
articles = j["articles"]

data, sentiment, compound = calculate_sentiment(articles)
save_articles(data, sentiment, compound)

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
A total of {total} business news articles was analysed. Out of that, {pos_score} had positive sentiment
({pos_pct:.1f}%), {neu_score} had neutral sentiment ({neu_pct:.1f}%), and {neg_score} had negative sentiment
({neg_pct:.1f}%). The average compound sentiment calculated was {average:+.3f} and the sentiment percentage range
calculated was {balance:+.1f}. {statement}.

These results describe the language in this news sample; they do not predict stock prices or establish investor
sentiment. The findings are limited to the retrieved articles and the text available through the API, which may be
truncated.
"""
