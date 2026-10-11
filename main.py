# import nltk
# nltk.download('vader_lexicon')

import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
sia = SentimentIntensityAnalyzer()

API_KEY = str(os.getenv("API_KEY"))
API_URL = "https://newsapi.org/v2/top-headlines"


def calculate_sentiment(headlines):
    refined = []
    scores = []
    comp_scores = []

    for a in headlines:
        publisher = a["source"]["name"]
        pub_date = datetime.fromisoformat(a["publishedAt"])
        title = a["title"]
        a_url = a["url"]

        description = "" if not a["description"] else a["description"]
        content = "" if not a["content"] else a["content"]

        refined.append([publisher, pub_date, title, a_url])

        text = f"{title}. {description} {content}"
        compound = sia.polarity_scores(text)["compound"]
        comp_scores.append(compound)

        if compound >= 0.05:
            scores.append("pos")
        elif compound <= -0.05:
            scores.append("neg")
        else:
            scores.append("neu")

    return refined, scores, comp_scores


r = requests.get(
    API_URL, headers={"X-Api-Key": API_KEY}, params={"category": "business"}, timeout=30
)
j = r.json()
articles = j["articles"]

data, sentiments, compound_scores = calculate_sentiment(articles)

pos_score = sentiments.count("pos")
neu_score = sentiments.count("neu")
neg_score = sentiments.count("neg")

total = len(sentiments)
pos_pct = pos_score / total * 100
neu_pct = neu_score / total * 100
neg_pct = neg_score / total * 100
average = sum(compound_scores) / total
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
