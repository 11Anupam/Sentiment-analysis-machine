"""
scraper.py — Reddit + Twitter/X data collection
Uses PRAW for Reddit (free API) and snscrape for Twitter (no auth needed)
Falls back to synthetic demo data if credentials not set
"""

import praw
import pandas as pd
import datetime
import random
import time
import os

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

def scrape_twitter(brand: str, limit: int = 100, days_back: int = 30) -> pd.DataFrame:
    """
    Scrape Twitter/X mentions using snscrape (no API key required).
    """
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
