"""
app.py — Brand Sentiment & Image Index Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import datetime
import warnings
warnings.filterwarnings("ignore")

from scraper     import collect_mentions
from sentiment   import analyze, top_keywords
from image_index import compute_brand_index, compute_all_indices, estimate_brand_equity_impact

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brand Image Index",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #e8e6f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111119;
    border-right: 1px solid #2a2a3a;
}
section[data-testid="stSidebar"] * {
    color: #c8c4d8 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

/* Index gauge card */
.index-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}

.index-number {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}

.index-label {
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8888aa;
    margin-top: 0.3rem;
}

/* Trend badges */
.badge-rising  { color: #00ff88; font-weight: 700; }
.badge-falling { color: #ff4466; font-weight: 700; }
.badge-stable  { color: #aaaacc; font-weight: 700; }

/* Demo mode banner */
.demo-banner {
    background: #1a1200;
    border: 1px solid #ffd700;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #ffd700;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* Section headers */
.section-title {
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #e94560;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}

/* Plotly chart backgrounds */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Scrollable table */
.dataframe { font-size: 0.82rem; }

button[kind="primary"] {
    background: #e94560 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#111119",
    font=dict(family="DM Mono, monospace", color="#c8c4d8", size=12),
    xaxis=dict(gridcolor="#1e1e30", linecolor="#2a2a3a", tickcolor="#2a2a3a"),
    yaxis=dict(gridcolor="#1e1e30", linecolor="#2a2a3a", tickcolor="#2a2a3a"),
    legend=dict(bgcolor="#0a0a0f", bordercolor="#2a2a3a"),
    margin=dict(l=40, r=20, t=40, b=40),
)

BRAND_COLORS = ["#e94560", "#00d4ff", "#7fff7f", "#ffaa00", "#bf5fff", "#ff7f50"]

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Session state ─────────────────────────────────────────────────────────────

if "df_raw"      not in st.session_state: st.session_state.df_raw      = None
if "df_analyzed" not in st.session_state: st.session_state.df_analyzed = None
if "indices"     not in st.session_state: st.session_state.indices     = None
if "brands"      not in st.session_state: st.session_state.brands      = []
if "is_demo"     not in st.session_state: st.session_state.is_demo     = True

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 Brand Image Index")
    st.markdown("---")

    st.markdown("**Brands to track**")
    brand_input = st.text_input(
        "Enter brands (comma-separated)",
        value="Apple, Tesla, Nike",
        label_visibility="collapsed",
    )
    brands = [b.strip() for b in brand_input.split(",") if b.strip()]

    st.markdown("**Data volume per brand**")
    mentions_per_brand = st.slider("Mentions to collect", 50, 300, 100, 25)

    st.markdown("**Lookback window**")
    days_back = st.slider("Days back", 7, 90, 30, 7)

    st.markdown("**Index weights**")
    with st.expander("Customise weights"):
        w_sent = st.slider("Sentiment",    0, 100, 40)
        w_vol  = st.slider("Volume",       0, 100, 30)
        w_mom  = st.slider("Momentum",     0, 100, 20)
        w_pos  = st.slider("Positivity",   0, 100, 10)
        total  = w_sent + w_vol + w_mom + w_pos
        if total != 100:
            st.warning(f"Weights sum to {total}. Will normalise automatically.")
        total  = max(total, 1)
        weights = {
            "sentiment":  w_sent / total,
            "volume":     w_vol  / total,
            "momentum":   w_mom  / total,
            "positivity": w_pos  / total,
        }

    st.markdown("**Valuation (optional)**")
    with st.expander("Brand equity estimate"):
        equity_brand  = st.selectbox("Brand", brands if brands else ["–"])
        equity_value  = st.number_input("Equity estimate (₹ Cr)", value=50000, step=1000)

    st.markdown("---")
    run_btn = st.button("🔍  Run Analysis", type="primary", use_container_width=True)


# ── Data collection & analysis ────────────────────────────────────────────────

if run_btn and brands:
    with st.spinner("Collecting mentions from Reddit & Twitter…"):
        df_raw = collect_mentions(
            brands,
            reddit_limit=mentions_per_brand // 2,
            twitter_limit=mentions_per_brand // 2,
            days_back=days_back,
        )

    with st.spinner("Running NLP sentiment pipeline…"):
        df_analyzed = analyze(df_raw, use_hf=False)

    with st.spinner("Computing Brand Image Index…"):
        indices = compute_all_indices(df_analyzed, brands, weights=weights)

    st.session_state.df_raw      = df_raw
    st.session_state.df_analyzed = df_analyzed
    st.session_state.indices     = indices
    st.session_state.brands      = brands
    st.session_state.is_demo     = "_demo" in df_analyzed.columns and df_analyzed["_demo"].any()


# ── Main dashboard ────────────────────────────────────────────────────────────

df       = st.session_state.df_analyzed
indices  = st.session_state.indices
brands   = st.session_state.brands
is_demo  = st.session_state.is_demo

if df is None:
    # Landing screen
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;'>
        <div style='font-size:3.5rem; font-weight:800; letter-spacing:-0.02em; color:#e8e6f0;'>
            Brand Image Index
        </div>
        <div style='font-size:1rem; color:#8888aa; margin-top:0.75rem; letter-spacing:0.1em;'>
            SENTIMENT INTELLIGENCE FOR BRAND VALUATION
        </div>
        <div style='margin-top:2.5rem; font-size:0.95rem; color:#c8c4d8; max-width:500px; margin-left:auto; margin-right:auto; line-height:1.8;'>
            Track Reddit &amp; Twitter sentiment in real time.<br>
            Convert brand perception into a 0–100 composite index.<br>
            Link public sentiment to brand equity estimates.
        </div>
        <div style='margin-top:3rem; font-size:0.8rem; color:#555577; letter-spacing:0.15em;'>
            ← Configure brands in the sidebar and click Run Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Demo banner
if is_demo:
    st.markdown("""
    <div class='demo-banner'>
    ⚠ <strong>Demo Mode</strong> — Showing synthetic data. 
    Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env vars to use live data.
    </div>
    """, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    f"<div class='section-title'>Brand Image Index — {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
    unsafe_allow_html=True,
)

# ── Index Cards (top row) ─────────────────────────────────────────────────────

cols = st.columns(len(brands))
for i, brand in enumerate(brands):
    row = indices[indices["Brand"] == brand]
    if row.empty:
        continue
    row = row.iloc[0]
    idx   = row["Image Index"]
    trend = row["Trend"]
    trend_icon = {"rising": "↑", "falling": "↓", "stable": "→"}.get(trend, "→")
    trend_css  = {"rising": "badge-rising", "falling": "badge-falling", "stable": "badge-stable"}.get(trend, "badge-stable")

    color = (
        "#00ff88" if idx >= 65 else
        "#ffaa00" if idx >= 45 else
        "#ff4466"
    )

    with cols[i]:
        st.markdown(f"""
        <div class='index-card'>
            <div style='font-size:1rem; font-weight:700; letter-spacing:0.1em; color:#8888aa; margin-bottom:0.5rem;'>
                {brand.upper()}
            </div>
            <div class='index-number' style='color:{color};'>{idx:.0f}</div>
            <div class='index-label'>Image Index</div>
            <div style='margin-top:0.6rem;'>
                <span class='{trend_css}'>{trend_icon} {trend.capitalize()}</span>
                <span style='color:#555577; margin-left:0.5rem; font-size:0.8rem;'>
                    {row["Total Mentions"]:,} mentions
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Trends", "☁ Word Cloud", "📋 Mentions", "💰 Valuation"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Comparison radar chart
    st.markdown("<div class='section-title'>Multi-Brand Comparison</div>", unsafe_allow_html=True)

    categories = ["Sentiment", "Volume", "Momentum", "Positivity"]
    fig_radar = go.Figure()
    for i, brand in enumerate(brands):
        row = indices[indices["Brand"] == brand]
        if row.empty:
            continue
        row = row.iloc[0]
        values = [
            row["Sentiment Score"],
            row["Volume Score"],
            row["Momentum Score"],
            row["Positivity Ratio"],
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=brand,
            line_color=BRAND_COLORS[i % len(BRAND_COLORS)],
            fillcolor=hex_to_rgba(BRAND_COLORS[i % len(BRAND_COLORS)], 0.15),
            opacity=0.85,
        ))

    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e1e30", linecolor="#2a2a3a", tickcolor="#2a2a3a"),
            angularaxis=dict(linecolor="#2a2a3a", gridcolor="#1e1e30"),
            bgcolor="#111119",
        ),
        showlegend=True,
        height=380,
        title=dict(text="Index Component Breakdown", font=dict(size=13, color="#c8c4d8")),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Summary table
    st.markdown("<div class='section-title'>Summary Table</div>", unsafe_allow_html=True)

    display_cols = ["Brand", "Image Index", "Sentiment Score", "Volume Score",
                    "Momentum Score", "Positivity Ratio", "Total Mentions", "Trend"]
    st.dataframe(
        indices[display_cols].style.format({
            "Image Index":      "{:.1f}",
            "Sentiment Score":  "{:.1f}",
            "Volume Score":     "{:.1f}",
            "Momentum Score":   "{:.1f}",
            "Positivity Ratio": "{:.1f}",
        }).background_gradient(subset=["Image Index"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True,
    )

    # Sentiment distribution bars
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Sentiment Distribution</div>", unsafe_allow_html=True)
    dist_data = []
    for brand in brands:
        bdf = df[df["brand"] == brand]
        total = len(bdf)
        if total == 0:
            continue
        dist_data.append({
            "Brand":    brand,
            "Positive": (bdf["sentiment_label"] == "positive").sum() / total * 100,
            "Neutral":  (bdf["sentiment_label"] == "neutral").sum()  / total * 100,
            "Negative": (bdf["sentiment_label"] == "negative").sum() / total * 100,
        })

    dist_df = pd.DataFrame(dist_data)
    if not dist_df.empty:
        fig_dist = go.Figure()
        for label, color in [("Positive", "#00ff88"), ("Neutral", "#8888aa"), ("Negative", "#ff4466")]:
            fig_dist.add_trace(go.Bar(
                name=label, x=dist_df["Brand"], y=dist_df[label],
                marker_color=color, opacity=0.85,
            ))
        fig_dist.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=280,
            title=dict(text="Sentiment Share (%)", font=dict(size=13, color="#c8c4d8")),
            yaxis_title="%",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from sentiment import daily_sentiment as _daily_sentiment

    st.markdown("<div class='section-title'>7-Day Rolling Sentiment</div>", unsafe_allow_html=True)

    fig_trend = go.Figure()
    for i, brand in enumerate(brands):
        daily = _daily_sentiment(df, brand)
        if daily.empty:
            continue
        color = BRAND_COLORS[i % len(BRAND_COLORS)]
        fig_trend.add_trace(go.Scatter(
            x=daily["day"],
            y=daily["rolling_sentiment"],
            name=brand,
            mode="lines",
            line=dict(color=color, width=2.5),
        ))
        # Shaded confidence band (±0.1)
        fig_trend.add_trace(go.Scatter(
            x=pd.concat([daily["day"], daily["day"][::-1]]),
            y=pd.concat([
                daily["rolling_sentiment"] + 0.08,
                (daily["rolling_sentiment"] - 0.08)[::-1],
            ]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.08),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig_trend.add_hline(y=0, line_dash="dot", line_color="#444466", annotation_text="Neutral")
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        title=dict(text="Sentiment Trend (7-day rolling avg)", font=dict(size=13, color="#c8c4d8")),
        yaxis_title="Sentiment Score",
    )
    fig_trend.update_yaxes(range=[-1, 1])
    st.plotly_chart(fig_trend, use_container_width=True)

    # Mention volume over time
    st.markdown("<div class='section-title'>Mention Volume Over Time</div>", unsafe_allow_html=True)

    fig_vol = go.Figure()
    for i, brand in enumerate(brands):
        daily = _daily_sentiment(df, brand)
        if daily.empty:
            continue
        color = BRAND_COLORS[i % len(BRAND_COLORS)]
        fig_vol.add_trace(go.Bar(
            x=daily["day"],
            y=daily["mention_count"],
            name=brand,
            marker_color=color,
            opacity=0.75,
        ))

    fig_vol.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=280,
        title=dict(text="Daily Mention Volume", font=dict(size=13, color="#c8c4d8")),
        yaxis_title="Mentions",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Image Index over time (rolling)
    st.markdown("<div class='section-title'>Image Index Over Time</div>", unsafe_allow_html=True)
    fig_idx_time = go.Figure()
    for i, brand in enumerate(brands):
        daily = _daily_sentiment(df, brand)
        if daily.empty:
            continue
        # Proxy rolling index from rolling_sentiment
        rolling_idx = ((daily["rolling_sentiment"] + 1) / 2 * 100).clip(0, 100)
        color = BRAND_COLORS[i % len(BRAND_COLORS)]
        fig_idx_time.add_trace(go.Scatter(
            x=daily["day"], y=rolling_idx,
            name=brand, mode="lines",
            line=dict(color=color, width=2.5),
        ))

    fig_idx_time.add_hline(y=50, line_dash="dot", line_color="#444466", annotation_text="Neutral (50)")
    fig_idx_time.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text="Rolling Image Index (sentiment component)", font=dict(size=13, color="#c8c4d8")),
        yaxis_title="Index (0–100)",
    )
    fig_idx_time.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_idx_time, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WORD CLOUD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Top Keywords by Sentiment</div>", unsafe_allow_html=True)

    col_brand, col_sent = st.columns(2)
    with col_brand:
        wc_brand = st.selectbox("Brand", brands, key="wc_brand")
    with col_sent:
        wc_sent = st.selectbox("Sentiment", ["positive", "negative", "neutral"], key="wc_sent")

    freq = top_keywords(df, wc_brand, sentiment=wc_sent, top_n=60)

    if not freq:
        st.info("Not enough data to generate word cloud.")
    else:
        color_maps = {
            "positive": "Greens",
            "negative": "Reds",
            "neutral":  "Blues",
        }

        wc = WordCloud(
            width=900,
            height=420,
            background_color="#0a0a0f",
            colormap=color_maps[wc_sent],
            max_words=60,
            collocations=False,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(freq)

        fig_wc, ax = plt.subplots(figsize=(11, 5))
        fig_wc.patch.set_facecolor("#0a0a0f")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

    # Top 15 words table alongside
    if freq:
        st.markdown("<div class='section-title' style='margin-top:1rem;'>Top 15 Keywords</div>", unsafe_allow_html=True)
        top15 = pd.DataFrame(
            list(freq.items())[:15],
            columns=["Keyword", "Frequency"],
        ).sort_values("Frequency", ascending=False).reset_index(drop=True)

        fig_kw = px.bar(
            top15, x="Frequency", y="Keyword", orientation="h",
            color="Frequency",
            color_continuous_scale=["#1a1a2e", "#e94560"] if wc_sent == "negative" else ["#1a1a2e", "#00ff88"],
        )
        fig_kw.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                             coloraxis_showscale=False,
                             title=dict(text=f"Top Keywords — {wc_brand} / {wc_sent}", font=dict(size=13, color="#c8c4d8")))
        fig_kw.update_traces(marker_line_width=0)
        st.plotly_chart(fig_kw, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MENTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Raw Mentions</div>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_brand = st.multiselect("Brand", brands, default=brands, key="f_brand")
    with col_f2:
        filter_sent  = st.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                      default=["positive", "neutral", "negative"], key="f_sent")
    with col_f3:
        filter_src   = st.multiselect("Source", df["source"].unique().tolist(),
                                      default=df["source"].unique().tolist(), key="f_src")

    filtered = df[
        df["brand"].isin(filter_brand) &
        df["sentiment_label"].isin(filter_sent) &
        df["source"].isin(filter_src)
    ].copy()

    filtered["date_str"]     = filtered["date"].dt.strftime("%d %b %Y %H:%M")
    filtered["score_str"]    = filtered["ensemble_score"].round(3).astype(str)

    display_df = filtered[["date_str", "brand", "source", "sentiment_label", "score_str", "text"]].copy()
    display_df.columns = ["Date", "Brand", "Source", "Sentiment", "Score", "Text"]
    display_df = display_df.sort_values("Date", ascending=False).reset_index(drop=True)

    # Colour sentiment column
    def color_sentiment(val):
        colors = {"positive": "color: #00ff88", "negative": "color: #ff4466", "neutral": "color: #8888aa"}
        return colors.get(val, "")

    st.dataframe(
        display_df.style.applymap(color_sentiment, subset=["Sentiment"]),
        use_container_width=True,
        height=450,
    )

    st.caption(f"Showing {len(display_df):,} of {len(df):,} total mentions")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>Brand Equity Impact Estimate</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#8888aa; font-size:0.88rem; margin-bottom:1.5rem; line-height:1.7;'>
    This is an <strong style='color:#ffaa00;'>illustrative model</strong>, not investment advice.
    The Image Index proxies public brand perception. Each point above/below 50 is mapped to a
    ±0.5% premium/discount on a user-supplied brand equity estimate.
    </div>
    """, unsafe_allow_html=True)

    val_brand = st.selectbox("Select brand", brands, key="val_brand_select",
                              index=brands.index(equity_brand) if equity_brand in brands else 0)
    val_equity = st.number_input("Base brand equity estimate (₹ Crores)", value=equity_value, step=1000)

    row = indices[indices["Brand"] == val_brand]
    if not row.empty:
        row = row.iloc[0]
        result = estimate_brand_equity_impact(row["Image Index"], val_equity)

        c1, c2, c3 = st.columns(3)
        c1.metric("Image Index", f"{row['Image Index']:.0f}/100")
        c2.metric("Sentiment Premium", f"{result['sentiment_premium_pct']:+.1f}%")
        c3.metric("Adjusted Equity", f"₹{result['adjusted_equity_cr']:,.0f} Cr",
                  delta=f"₹{result['impact_cr']:+,.0f} Cr")

        # Waterfall chart
        fig_wf = go.Figure(go.Waterfall(
            name="Equity Build",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Base Equity", "Sentiment Adj.", "Adjusted Equity"],
            y=[result["base_equity_cr"], result["impact_cr"], result["adjusted_equity_cr"]],
            text=[f"₹{result['base_equity_cr']:,.0f}Cr",
                  f"₹{result['impact_cr']:+,.0f}Cr",
                  f"₹{result['adjusted_equity_cr']:,.0f}Cr"],
            textposition="outside",
            connector=dict(line=dict(color="#2a2a3a", width=1.5)),
            increasing=dict(marker_color="#00ff88"),
            decreasing=dict(marker_color="#ff4466"),
            totals=dict(marker_color="#00d4ff"),
        ))
        fig_wf.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            title=dict(text=f"Brand Equity Waterfall — {val_brand}", font=dict(size=13, color="#c8c4d8")),
            yaxis_title="₹ Crores",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Sensitivity table
        st.markdown("<div class='section-title' style='margin-top:1rem;'>Sensitivity Table</div>", unsafe_allow_html=True)
        sens_rows = []
        for scenario, idx_val in [
            ("Bear (Index 30)", 30),
            ("Base (Index 50)", 50),
            ("Current Index",   row["Image Index"]),
            ("Bull (Index 70)", 70),
            ("Euphoria (Index 85)", 85),
        ]:
            r = estimate_brand_equity_impact(idx_val, val_equity)
            sens_rows.append({
                "Scenario":          scenario,
                "Image Index":       f"{idx_val:.0f}",
                "Premium":           f"{r['sentiment_premium_pct']:+.1f}%",
                "Adjusted Equity":   f"₹{r['adjusted_equity_cr']:,.0f} Cr",
                "Delta":             f"₹{r['impact_cr']:+,.0f} Cr",
            })

        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    ---
    <div style='color:#444466; font-size:0.78rem;'>
    Model assumption: each 1pt Image Index deviation from 50 = 0.5% brand equity premium/discount.
    Adjust base equity to reflect analyst estimates, Interbrand / BrandZ valuations, or proprietary models.
    </div>
    """, unsafe_allow_html=True)
