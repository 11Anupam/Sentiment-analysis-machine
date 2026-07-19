🚀 Brand Sentiment & Image Index Dashboard

A production-ready analytics application that converts unstructured social and news data into a unified Brand Image Index (0–100) — enabling real-time tracking of brand perception, sentiment dynamics, and market positioning.

🎯 Why This Matters

Most sentiment tools stop at “positive/negative.”
This system goes further by combining sentiment + attention (volume) + trend (momentum) into a single decision-making metric.

👉 Built for marketing strategy, brand analytics, and investment insights.

🧠 Core Architecture

**Pipeline Design:
**
Data Collection → NLP Processing → Index Engine → Dashboard
Data Sources
Reddit (PRAW API)
Twitter/X (snscrape)
News (RSS feeds)
NLP Engine
VADER + TextBlob (ensemble scoring)
Keyword extraction for insights
Extendable to transformer models
📊 Image Index Model
Image Index = (0.5 × Sentiment) + (0.3 × Volume) + (0.2 × Momentum)
Sentiment → Perception
Volume → Awareness
Momentum → Direction

✨ Key Features:

📌 Multi-brand comparison
📈 Sentiment & volume trend analysis
☁️ Word cloud (positive vs negative signals)
📊 Real-time Image Index scoring
🔍 Raw mention-level insights
🧪 Demo mode (runs without API keys)

📂 Project Structure

brand_sentiment_app/
├── app.py              # Streamlit dashboard
├── scraper.py          # Data collection layer
├── sentiment.py        # NLP pipeline
├── image_index.py      # Scoring engine
├── requirements.txt
└── Brand_Sentiment_Image_Index.ipynb


⚙️ Getting Started
Run Locally
git clone https://github.com/11Anupam/Sentiment-analysis-machine.git
cd Sentiment-analysis-machine
pip install -r requirements.txt
streamlit run app.py

**
Run on Colab**
Open notebook
Run cells (auto setup)
Access via ngrok link
Deploy (Streamlit Cloud)
Connect GitHub repo
Select app.py
Add optional API keys
Deploy in ~2 minutes



🔑 Environment Variables (Optional)

REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
XQUIK_API_KEY=your_key
XQUIK_API_BASE_URL=https://xquik.com/api/v1
XQUIK_TWEETS_CSV=path/to/reviewed-tweets.csv

`XQUIK_API_KEY` is optional. When set, the Twitter/X collector searches recent
posts through the Xquik REST API before falling back to live collection or demo
data. `XQUIK_API_BASE_URL` only needs to be set for compatible custom deployments.

`XQUIK_TWEETS_CSV` is an optional offline override for a reviewed CSV file. It is
checked before the API. Supported columns include `text`, `content`, `tweet`,
`rawContent`, `date`, `created_at`, `url`, `tweet_url`, `score`, `likes`,
`brand`, `query`, and `keyword`.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.



🧩 Tech Stack

Python
Streamlit
PRAW
snscrape
feedparser
VADER, TextBlob


📈 Use Cases
Brand monitoring & reputation tracking
Marketing performance analysis
Competitive benchmarking
Sentiment-driven investment signals


⚠️ Limitations
Twitter scraping may face rate limits
Data quality depends on source noise
Demo mode used if APIs unavailable


🔮 Roadmap
Transformer-based sentiment (BERT)
Real-time streaming pipeline
Predictive brand scoring
Financial market correlation models

🤝 Contribution

Open to improvements and extensions. Fork, build, and raise a PR.

📄 License

MIT License
