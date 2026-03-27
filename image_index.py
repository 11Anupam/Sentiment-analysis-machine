"""
image_index.py — Brand Image Index calculation engine

Image Index (0–100) is a composite score combining:
    1. Sentiment Score     (40%) — avg ensemble sentiment, scaled
    2. Volume Score        (30%) — mention volume relative to baseline
    3. Momentum Score      (20%) — direction of 7-day rolling sentiment
    4. Positivity Ratio    (10%) — share of positive vs negative mentions

Higher = stronger brand perception.
"""

import pandas as pd
import numpy as np
from sentiment import daily_sentiment


# ── Scaling helpers ──────────────────────────────────────────────────────────

def _scale_to_100(value: float, min_val: float, max_val: float) -> float:
    """Min-max scale a value to 0–100."""
    if max_val == min_val:
        return 50.0
    scaled = (value - min_val) / (max_val - min_val) * 100
    return float(np.clip(scaled, 0, 100))


def _sentiment_to_score(avg_sentiment: float) -> float:
    """
    Convert avg sentiment (-1 to +1) → 0–100 score.
    Neutral (0) → 50, fully positive (+1) → 100, fully negative (-1) → 0.
    """
    return float(np.clip((avg_sentiment + 1) / 2 * 100, 0, 100))


# ── Per-brand index ──────────────────────────────────────────────────────────

def compute_brand_index(
    df: pd.DataFrame,
    brand: str,
    weights: dict = None,
) -> dict:
    """
    Compute the Image Index for a single brand.

    Returns a dict:
    {
        "brand":           str,
        "image_index":     float (0–100),
        "sentiment_score": float (0–100),
        "volume_score":    float (0–100),
        "momentum_score":  float (0–100),
        "positivity_ratio":float (0–100),
        "mention_count":   int,
        "avg_sentiment":   float (-1 to +1),
        "trend":           "rising" | "falling" | "stable",
        "daily":           pd.DataFrame,
    }
    """
    if weights is None:
        weights = {
            "sentiment": 0.40,
            "volume":    0.30,
            "momentum":  0.20,
            "positivity":0.10,
        }

    brand_df = df[df["brand"] == brand]
    if brand_df.empty:
        return _empty_result(brand)

    daily = daily_sentiment(df, brand)

    if daily.empty:
        return _empty_result(brand)

    # ── 1. Sentiment Score ───────────────────────────────────────────────
    avg_sent      = daily["avg_sentiment"].mean()
    sentiment_score = _sentiment_to_score(avg_sent)

    # ── 2. Volume Score ──────────────────────────────────────────────────
    # Score relative to expected: 50 = average, 100 = 2× expected
    total_mentions = int(brand_df.shape[0])
    days_span      = max((daily["day"].max() - daily["day"].min()).days, 1)
    avg_daily_vol  = total_mentions / days_span
    # Normalise: 10 mentions/day = 50, 30+ = 100
    volume_score = _scale_to_100(avg_daily_vol, 0, 30)

    # ── 3. Momentum Score ────────────────────────────────────────────────
    # Compare last 7 days rolling avg to first 7 days
    if len(daily) >= 7:
        early = daily["rolling_sentiment"].iloc[:7].mean()
        late  = daily["rolling_sentiment"].iloc[-7:].mean()
        delta = late - early  # range roughly -2 to +2
    else:
        delta = daily["avg_sentiment"].diff().mean() or 0.0

    momentum_score = _scale_to_100(delta, -0.5, 0.5)

    # ── 4. Positivity Ratio ──────────────────────────────────────────────
    pos = (brand_df["sentiment_label"] == "positive").sum()
    neg = (brand_df["sentiment_label"] == "negative").sum()
    total = pos + neg
    pos_ratio = (pos / total) if total > 0 else 0.5
    positivity_score = pos_ratio * 100

    # ── Composite Image Index ────────────────────────────────────────────
    image_index = (
        weights["sentiment"]  * sentiment_score  +
        weights["volume"]     * volume_score      +
        weights["momentum"]   * momentum_score    +
        weights["positivity"] * positivity_score
    )
    image_index = float(np.clip(image_index, 0, 100))

    # ── Trend label ──────────────────────────────────────────────────────
    if delta > 0.05:
        trend = "rising"
    elif delta < -0.05:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "brand":            brand,
        "image_index":      round(image_index, 1),
        "sentiment_score":  round(sentiment_score, 1),
        "volume_score":     round(volume_score, 1),
        "momentum_score":   round(momentum_score, 1),
        "positivity_ratio": round(positivity_score, 1),
        "mention_count":    total_mentions,
        "avg_sentiment":    round(float(avg_sent), 4),
        "trend":            trend,
        "daily":            daily,
    }


def _empty_result(brand: str) -> dict:
    return {
        "brand":            brand,
        "image_index":      0.0,
        "sentiment_score":  0.0,
        "volume_score":     0.0,
        "momentum_score":   0.0,
        "positivity_ratio": 0.0,
        "mention_count":    0,
        "avg_sentiment":    0.0,
        "trend":            "stable",
        "daily":            pd.DataFrame(),
    }


# ── Multi-brand comparison ───────────────────────────────────────────────────

def compute_all_indices(
    df: pd.DataFrame,
    brands: list,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute Image Index for multiple brands.
    Returns a summary DataFrame (one row per brand).
    """
    results = []
    for brand in brands:
        r = compute_brand_index(df, brand, weights=weights)
        results.append({
            "Brand":             r["brand"],
            "Image Index":       r["image_index"],
            "Sentiment Score":   r["sentiment_score"],
            "Volume Score":      r["volume_score"],
            "Momentum Score":    r["momentum_score"],
            "Positivity Ratio":  r["positivity_ratio"],
            "Total Mentions":    r["mention_count"],
            "Avg Sentiment":     r["avg_sentiment"],
            "Trend":             r["trend"],
        })

    return pd.DataFrame(results).sort_values("Image Index", ascending=False).reset_index(drop=True)


# ── Valuation proxy ──────────────────────────────────────────────────────────

def estimate_brand_equity_impact(image_index: float, base_equity_cr: float) -> dict:
    """
    Rough brand equity impact model (illustrative, not investment advice).

    Assumes:
    - Image Index 50 = neutral / no impact on brand equity
    - Each point above 50 adds ~0.5% brand equity premium
    - Each point below 50 subtracts ~0.5% brand equity discount

    base_equity_cr: estimated brand equity in INR crores
    """
    delta_pct = (image_index - 50) * 0.5  # % change
    adjusted  = base_equity_cr * (1 + delta_pct / 100)

    return {
        "base_equity_cr":     round(base_equity_cr, 2),
        "sentiment_premium_pct": round(delta_pct, 2),
        "adjusted_equity_cr": round(adjusted, 2),
        "impact_cr":          round(adjusted - base_equity_cr, 2),
    }
