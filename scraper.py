"""
scraper.py — Reddit + Twitter/X data collection
Uses PRAW for Reddit (free API) and snscrape for Twitter (no auth needed)
Falls back to synthetic demo data if credentials not set
"""

import datetime
import json
import os
import random
import time
from typing import Optional, Tuple
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
import praw

# ── Reddit scraper ──────────────────────────────────────────────────────────

def get_reddit_client():
    """Initialize Reddit client. Uses env vars or falls back to demo mode."""
    client_id     = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent    = "BrandSentimentBot/1.0"

    if not client_id or not client_secret:
        return None

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        return reddit
    except Exception:
        return None


def scrape_reddit(brand: str, limit: int = 100, time_filter: str = "month") -> pd.DataFrame:
    """
    Scrape Reddit posts and top comments mentioning `brand`.
    time_filter: 'day', 'week', 'month', 'year', 'all'
    """
    reddit = get_reddit_client()

    if reddit is None:
        return _demo_data(brand, limit, source="reddit")

    records = []
    try:
        for submission in reddit.subreddit("all").search(
            brand, sort="new", time_filter=time_filter, limit=limit
        ):
            records.append({
                "date":   datetime.datetime.fromtimestamp(submission.created_utc),
                "text":   submission.title + " " + (submission.selftext or ""),
                "source": "reddit",
                "brand":  brand,
                "url":    f"https://reddit.com{submission.permalink}",
                "score":  submission.score,
            })
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:5]:
                records.append({
                    "date":   datetime.datetime.fromtimestamp(comment.created_utc),
                    "text":   comment.body,
                    "source": "reddit_comment",
                    "brand":  brand,
                    "url":    f"https://reddit.com{submission.permalink}",
                    "score":  comment.score,
                })
    except Exception as e:
        print(f"[scraper] Reddit error: {e} — using demo data")
        return _demo_data(brand, limit, source="reddit")

    if not records:
        return _demo_data(brand, limit, source="reddit")

    return pd.DataFrame(records)


# ── Twitter / X scraper ─────────────────────────────────────────────────────

CSV_COLUMN_ALIASES = {
    "text": ("text", "content", "tweet", "rawContent", "raw_content", "full_text"),
    "date": ("date", "created_at", "createdAt", "timestamp", "time"),
    "url": ("url", "tweet_url", "link", "permalink"),
    "score": ("score", "likeCount", "like_count", "likes", "favorite_count"),
    "brand": ("brand", "query", "keyword", "topic"),
}

XQUIK_API_BASE_URL = "https://xquik.com/api/v1"
XQUIK_API_CONTRACT = "2026-04-29"
XQUIK_REQUEST_TIMEOUT_SECONDS = 70


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def urlopen(request, timeout):
    opener = build_opener(RejectRedirects())
    return opener.open(request, timeout=timeout)


def _pick_column(df: pd.DataFrame, aliases: Tuple[str, ...]) -> Optional[str]:
    for name in aliases:
        if name in df.columns:
            return name
    lower_map = {str(col).lower(): col for col in df.columns}
    for name in aliases:
        found = lower_map.get(name.lower())
        if found is not None:
            return found
    return None


def search_xquik_mentions(
    brand: str,
    limit: int = 100,
    days_back: int = 30,
) -> Optional[pd.DataFrame]:
    """Fetch recent X mentions through the optional Xquik REST integration."""
    api_key = os.getenv("XQUIK_API_KEY", "").strip()
    if not api_key or limit <= 0:
        return None

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=max(days_back, 0))
    since_time = cutoff.isoformat().replace("+00:00", "Z")
    query = urlencode({
        "q": brand,
        "queryType": "Latest",
        "limit": min(limit, 200),
        "sinceTime": since_time,
    })
    request = Request(
        f"{XQUIK_API_BASE_URL}/x/tweets/search?{query}",
        headers={
            "Accept": "application/json",
            "x-api-key": api_key,
            "xquik-api-contract": XQUIK_API_CONTRACT,
        },
    )

    try:
        with urlopen(request, timeout=XQUIK_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (OSError, ValueError) as error:
        print(
            f"[scraper] Xquik API error: {error} - falling back to the next source"
        )
        return None

    tweets = payload.get("tweets") if isinstance(payload, dict) else None
    if not isinstance(tweets, list):
        return None

    records = []
    for tweet in tweets[:limit]:
        if not isinstance(tweet, dict):
            continue
        text = tweet.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        tweet_id = tweet.get("id")
        url = tweet.get("url")
        if not isinstance(url, str) and tweet_id is not None:
            url = f"https://x.com/i/status/{tweet_id}"
        records.append({
            "date": _parse_xquik_timestamp(
                tweet.get("created")
                or tweet.get("createdAt")
                or tweet.get("created_at")
            ),
            "text": text,
            "source": "xquik_api",
            "brand": brand,
            "url": url if isinstance(url, str) else "",
            "score": tweet.get("likeCount", tweet.get("like_count", 0)) or 0,
        })

    if not records:
        return None

    frame = pd.DataFrame(records)
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    recent = parsed_dates.notna() & (parsed_dates >= pd.Timestamp(cutoff))
    frame = frame.loc[recent].copy()
    if frame.empty:
        return None
    frame["date"] = parsed_dates.loc[recent].dt.tz_convert(None)
    return frame


def _parse_xquik_timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", utc=True)


def load_xquik_csv_mentions(
    brand: str,
    limit: int = 100,
    days_back: int = 30,
) -> Optional[pd.DataFrame]:
    """Load reviewed X/Twitter mentions from an optional CSV override."""
    csv_path = os.getenv("XQUIK_TWEETS_CSV", "").strip()
    if not csv_path:
        return None

    try:
        raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[scraper] Xquik CSV error: {e} - falling back to live Twitter/X collection")
        return None

    text_col = _pick_column(raw, CSV_COLUMN_ALIASES["text"])
    if text_col is None:
        print("[scraper] Xquik CSV missing a text/content column - falling back to live Twitter/X collection")
        return None

    date_col = _pick_column(raw, CSV_COLUMN_ALIASES["date"])
    url_col = _pick_column(raw, CSV_COLUMN_ALIASES["url"])
    score_col = _pick_column(raw, CSV_COLUMN_ALIASES["score"])
    brand_col = _pick_column(raw, CSV_COLUMN_ALIASES["brand"])

    df = raw.copy()
    if brand_col is not None:
        brand_filter = df[brand_col].astype(str).str.casefold() == brand.casefold()
    else:
        brand_filter = df[text_col].astype(str).str.contains(brand, case=False, na=False, regex=False)
    df = df.loc[brand_filter].head(limit)

    if df.empty:
        return None

    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        dates = pd.Series([datetime.datetime.now()] * len(df), index=df.index)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    date_filter = dates.isna() | (dates >= cutoff)
    df = df.loc[date_filter]
    dates = dates.loc[date_filter].fillna(pd.Timestamp.now())

    if df.empty:
        return None

    return pd.DataFrame({
        "date": dates,
        "text": df[text_col].astype(str),
        "source": "xquik_csv",
        "brand": brand,
        "url": df[url_col].astype(str) if url_col is not None else "",
        "score": pd.to_numeric(df[score_col], errors="coerce").fillna(0) if score_col is not None else 0,
    }).head(limit)


def scrape_twitter(brand: str, limit: int = 100, days_back: int = 30) -> pd.DataFrame:
    """
    Scrape Twitter/X mentions using snscrape (no API key required).
    """
    csv_mentions = load_xquik_csv_mentions(brand, limit=limit, days_back=days_back)
    if csv_mentions is not None:
        return csv_mentions

    api_mentions = search_xquik_mentions(brand, limit=limit, days_back=days_back)
    if api_mentions is not None:
        return api_mentions

    try:
        import snscrape.modules.twitter as sntwitter

        since = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{brand} lang:en since:{since}"

        records = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            records.append({
                "date":   tweet.date.replace(tzinfo=None),
                "text":   tweet.rawContent,
                "source": "twitter",
                "brand":  brand,
                "url":    tweet.url,
                "score":  tweet.likeCount or 0,
            })
            time.sleep(0.05)

        if not records:
            return _demo_data(brand, limit, source="twitter")

        return pd.DataFrame(records)

    except Exception as e:
        print(f"[scraper] Twitter fallback — demo data ({e})")
        return _demo_data(brand, limit, source="twitter")


# ── Combined collector ───────────────────────────────────────────────────────

def collect_mentions(
    brands: list,
    reddit_limit: int = 80,
    twitter_limit: int = 80,
    days_back: int = 30,
) -> pd.DataFrame:
    """Collect mentions for all brands from Reddit + Twitter."""
    all_dfs = []
    for brand in brands:
        reddit_df  = scrape_reddit(brand, limit=reddit_limit,  time_filter="month")
        twitter_df = scrape_twitter(brand, limit=twitter_limit, days_back=days_back)
        all_dfs.extend([reddit_df, twitter_df])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


# ── Demo / synthetic data ────────────────────────────────────────────────────

DEMO_TEMPLATES = {
    "positive": [
        "{brand} just blew my mind with their latest product. Absolutely love it.",
        "Have been using {brand} for months — customer service is top notch.",
        "{brand}'s new launch is genuinely impressive. Strong buy signal.",
        "Nobody does it like {brand}. Market leader for a reason.",
        "Upgraded to {brand} premium. Worth every rupee.",
        "{brand} just dropped something incredible. The hype is real.",
        "My experience with {brand} support was amazing — resolved in 10 mins.",
        "{brand} is undervalued right now. Sentiment is turning positive fast.",
        "The brand recall for {brand} is insane. Everyone's talking about it.",
        "{brand} partnership announcement is genius marketing. Big brand equity move.",
        "Switched to {brand} last month. Never going back. Quality is unreal.",
        "{brand} keeps delivering. Consistent, trustworthy, premium.",
    ],
    "negative": [
        "{brand} customer service is a nightmare. Three tickets, zero resolution.",
        "Overpriced and underdelivered — that's {brand} in a nutshell.",
        "{brand} app keeps crashing. Switching to a competitor.",
        "{brand} said something tone-deaf again. PR disaster incoming.",
        "Quality has dropped significantly at {brand}. Old {brand} was better.",
        "{brand} ad campaign feels desperate. Something is off at the top.",
        "If the data breach rumours about {brand} are true, huge trust issue.",
        "{brand} raised prices again. They clearly don't value loyal customers.",
        "Returned my {brand} purchase. Not impressed. Overhyped.",
        "{brand} layoffs signal deeper problems. Sentiment tanking.",
        "Worst unboxing experience ever. {brand} has lost the plot.",
        "{brand} keeps overpromising and underdelivering. Getting old.",
    ],
    "neutral": [
        "{brand} announced Q3 earnings. Numbers came in line with expectations.",
        "Anyone tried the new {brand} product? Curious what people think.",
        "{brand} expanding to new markets. Standard growth play.",
        "Comparing {brand} vs competitors — both have pros and cons.",
        "{brand} hired a new CMO. Early to say if this changes anything.",
        "Saw a {brand} ad on Instagram. Not sure how I feel about it yet.",
        "{brand} is doing a brand refresh apparently. Let's see.",
        "Just read an analyst note on {brand}. Mixed views across the board.",
        "{brand} partnered with an NGO for a sustainability push.",
        "{brand} opened a new store near me. Haven't visited yet.",
        "{brand} trending on social today. Not sure why.",
        "Decent quarter for {brand}. Nothing spectacular, nothing terrible.",
    ],
}


def _demo_data(brand: str, limit: int, source: str) -> pd.DataFrame:
    """Generate realistic synthetic data for demo/testing."""
    random.seed(hash(brand + source) % 99991)
    records = []

    # Slightly different sentiment profile per brand for realism
    brand_bias = {
        "Apple": (0.55, 0.20, 0.25),
        "Tesla": (0.40, 0.38, 0.22),
        "Nike":  (0.50, 0.22, 0.28),
    }.get(brand, (0.45, 0.27, 0.28))  # default

    sentiment_dist = [
        ("positive", brand_bias[0]),
        ("negative", brand_bias[1]),
        ("neutral",  brand_bias[2]),
    ]

    end_date   = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)

    for _ in range(limit):
        r   = random.random()
        cum = 0.0
        chosen = "neutral"
        for label, prob in sentiment_dist:
            cum += prob
            if r < cum:
                chosen = label
                break

        template = random.choice(DEMO_TEMPLATES[chosen])
        text     = template.replace("{brand}", brand)

        delta        = end_date - start_date
        rand_seconds = random.randint(0, int(delta.total_seconds()))
        date         = start_date + datetime.timedelta(seconds=rand_seconds)

        records.append({
            "date":   date,
            "text":   text,
            "source": source,
            "brand":  brand,
            "url":    f"https://example.com/{source}/{random.randint(10000, 99999)}",
            "score":  random.randint(0, 500),
            "_demo":  True,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
