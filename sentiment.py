"""
sentiment.py — NLP sentiment pipeline
Uses VADER (fast, social-media optimised) + TextBlob (polarity/subjectivity)
Optional: HuggingFace transformer for higher accuracy (slower)
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings("ignore")

# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text normalisation for NLP."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                     # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # keep hashtag words
    text = re.sub(r"[^\w\s!?.,'-]", " ", text)           # strip special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── VADER sentiment ──────────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()

def vader_scores(text: str) -> dict:
    """Returns VADER compound, pos, neu, neg scores."""
    scores = _vader.polarity_scores(clean_text(text))
    return scores


def vader_label(compound: float) -> str:
    """Classify compound score into sentiment label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


# ── TextBlob sentiment ───────────────────────────────────────────────────────

def textblob_scores(text: str) -> tuple:
    """Returns (polarity, subjectivity) from TextBlob."""
    blob = TextBlob(clean_text(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# ── HuggingFace (optional) ───────────────────────────────────────────────────

def hf_sentiment(texts: list) -> list:
    """
    Run a HuggingFace pipeline for more accurate sentiment.
    Only called if transformers is installed.
    Returns list of dicts: [{"label": "POSITIVE", "score": 0.98}, ...]
    """
    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        results = pipe(texts, batch_size=16)
        return results
    except ImportError:
        return []
    except Exception as e:
        print(f"[sentiment] HF pipeline error: {e}")
        return []


# ── Main analysis function ───────────────────────────────────────────────────

def analyze(df: pd.DataFrame, use_hf: bool = False) -> pd.DataFrame:
    """
    Run full sentiment analysis on a DataFrame with a 'text' column.
    Adds columns:
        clean_text, vader_compound, vader_pos, vader_neu, vader_neg,
        tb_polarity, tb_subjectivity, sentiment_label,
        ensemble_score  (weighted blend of VADER + TextBlob)
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # VADER
    vader_results = df["clean_text"].apply(lambda t: _vader.polarity_scores(t))
    df["vader_compound"] = vader_results.apply(lambda x: x["compound"])
    df["vader_pos"]      = vader_results.apply(lambda x: x["pos"])
    df["vader_neu"]      = vader_results.apply(lambda x: x["neu"])
    df["vader_neg"]      = vader_results.apply(lambda x: x["neg"])

    # TextBlob
    tb_results = df["clean_text"].apply(textblob_scores)
    df["tb_polarity"]     = tb_results.apply(lambda x: x[0])
    df["tb_subjectivity"] = tb_results.apply(lambda x: x[1])

    # HuggingFace (optional)
    if use_hf:
        hf_results = hf_sentiment(df["clean_text"].tolist())
        if hf_results:
            hf_scores = [
                r["score"] if r["label"] == "POSITIVE" else -r["score"]
                for r in hf_results
            ]
            df["hf_score"] = hf_scores
            # Ensemble: 40% VADER + 30% TextBlob + 30% HF
            df["ensemble_score"] = (
                0.40 * df["vader_compound"] +
                0.30 * df["tb_polarity"] +
                0.30 * df["hf_score"]
            )
        else:
            df["ensemble_score"] = (
                0.60 * df["vader_compound"] + 0.40 * df["tb_polarity"]
            )
    else:
        # Ensemble: 60% VADER + 40% TextBlob
        df["ensemble_score"] = (
            0.60 * df["vader_compound"] + 0.40 * df["tb_polarity"]
        )

    # Final label from ensemble score
    df["sentiment_label"] = df["ensemble_score"].apply(
        lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
    )

    return df


# ── Aggregation helpers ──────────────────────────────────────────────────────

def daily_sentiment(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Aggregate sentiment scores by day for a given brand."""
    brand_df = df[df["brand"] == brand].copy()
    brand_df["day"] = brand_df["date"].dt.date

    daily = brand_df.groupby("day").agg(
        avg_sentiment=("ensemble_score", "mean"),
        mention_count=("text",           "count"),
        pos_count    =("sentiment_label", lambda x: (x == "positive").sum()),
        neg_count    =("sentiment_label", lambda x: (x == "negative").sum()),
        neu_count    =("sentiment_label", lambda x: (x == "neutral").sum()),
    ).reset_index()

    daily["day"] = pd.to_datetime(daily["day"])
    daily["pos_ratio"] = daily["pos_count"] / daily["mention_count"]
    daily["neg_ratio"] = daily["neg_count"] / daily["mention_count"]

    # 7-day rolling average
    daily = daily.sort_values("day")
    daily["rolling_sentiment"] = daily["avg_sentiment"].rolling(7, min_periods=1).mean()
    daily["rolling_mentions"]  = daily["mention_count"].rolling(7, min_periods=1).mean()

    return daily


def top_keywords(df: pd.DataFrame, brand: str, sentiment: str = "positive", top_n: int = 30) -> dict:
    """
    Extract top keywords from mentions of a specific sentiment.
    Returns {word: frequency} dict for word cloud.
    """
    from collections import Counter

    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","was","are","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "that","this","it","its","i","me","my","we","our","you","your","he",
        "she","they","their","them","what","which","who","whom","when","where",
        "why","how","all","each","both","few","more","most","other","some",
        "such","no","not","only","same","so","than","too","very","just","also",
        "said","says","like","one","two","get","got","go","going","know","see",
        "think","make","made","way","even","well","back","first","come","good",
        "new","s","t","re","ve","ll","m","d","brand",
    }

    subset = df[(df["brand"] == brand) & (df["sentiment_label"] == sentiment)]
    words  = []
    for text in subset["clean_text"]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words.extend([w for w in tokens if w not in STOPWORDS and w != brand.lower()])

    freq = Counter(words).most_common(top_n)
    return dict(freq)
