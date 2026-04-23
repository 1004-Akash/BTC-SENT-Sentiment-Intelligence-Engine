 ```
██████╗ ████████╗ ██████╗    ███████╗███████╗███╗   ██╗████████╗
██╔══██╗╚══██╔══╝██╔════╝    ██╔════╝██╔════╝████╗  ██║╚══██╔══╝
██████╔╝   ██║   ██║         ███████╗█████╗  ██╔██╗ ██║   ██║   
██╔══██╗   ██║   ██║         ╚════██║██╔══╝  ██║╚██╗██║   ██║   
██████╔╝   ██║   ╚██████╗    ███████║███████╗██║ ╚████║   ██║   
╚═════╝    ╚═╝    ╚═════╝    ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
```
> **BTC-SENT · V1.1.0 RELEASE** — Sentiment Intelligence Engine

# Bitcoin Sentiment Intelligence Engine
### Fear & Greed × Hyperliquid Trader Performance

A high-performance data science pipeline correlating market sentiment signals with real-world trader outcomes from Hyperliquid. This project uncovers the emotional patterns that drive market alpha and provides a predictive framework for trade profitability.

![Python](https://img.shields.io/badge/Python-3.x-F5A623?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-00D68F?style=flat-square&logo=scikit-learn&logoColor=white)
![Model](https://img.shields.io/badge/Model-HistGradientBoosting-5B9CF6?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-95.8%25-FF4D4D?style=flat-square)
![Trades](https://img.shields.io/badge/Trades-211k%2B-F5A623?style=flat-square)
![Assessment](https://img.shields.io/badge/Primetrade.ai-AI%20Intern-A78BFA?style=flat-square)

---

## 📊 At a Glance

| Metric | Value | Description |
|:---|:---|:---|
| **Predictive Accuracy** | **95.8%** | HistGBC Ensemble on 80/20 split |
| **Trades Analyzed** | **211k+** | Hyperliquid historical dataset |
| **Sentiment Phases** | **5** | Extreme Fear → Extreme Greed |

---

## 🌡️ Sentiment Spectrum

```
[ EXTREME FEAR ]──[ FEAR ]──[ NEUTRAL ]──[ GREED ]──[ EXTREME GREED ]
      0–25          25–45      45–55       55–75          75–100
```

*Index Range: 0 (Max Fear) → 100 (Max Greed)*

---

## 🔍 Visual Analysis Gallery

### 01. Profitability vs. Sentiment Phase
![Avg PnL by Sentiment](avg_pnl_sentiment.png)
Traders achieved the highest average PnL during **Extreme Greed** phases, primarily through counter-trend selling at euphoria peaks.

### 02. Win Rate Probabilities
![Win Rate by Sentiment](win_rate_by_sentiment.png)
Win rates vary significantly across sentiment zones, with "Fear" and "Neutral" phases showing tighter execution windows compared to high-volatility "Extreme" phases.

### 03. PnL Distribution & Variance
![PnL Distribution](pnl_distribution.png)
Analyzing the spread of trade outcomes. Sentiment intensity correlates with a wider distribution of PnL, highlighting both increased risk and reward.

### 04. Sentiment Intensity Correlation
![PnL vs Sentiment Value](pnl_vs_sentiment_value.png)
A regression analysis of 211k+ trades showing how individual profitability scales across the 0-100 Fear & Greed spectrum.

---

## 🧠 ML Model — Intelligence Layer

### Training Results

| Parameter | Value |
|:---|:---|
| **Model** | `HistGradientBoostingClassifier` |
| **Final Accuracy** | **95.80%** |
| **Split** | 80% Train / 20% Test |
| **Target** | Trade profitability (binary) |

```
ACCURACY  ████████████████████████████████████████████████░░  95.8%
```

### Feature Engineering Matrix

| Feature Category | Description | Impact |
|:---|:---|:---:|
| **Price Relativity** | Execution price vs. daily average for each specific coin | 🔴 CRITICAL |
| **Sentiment Momentum** | MA3 moving averages and daily delta of Fear & Greed values | 🟡 HIGH |
| **Categorical Meta** | Account IDs and asset encodings | 🟡 HIGH |
| **Time-Based** | Hour of day and day of week execution patterns | 🔵 MED |

---

## 📁 Project Structure
```
btc-sent/
├── analyze_and_train.py      # Master pipeline — ingestion, model training & metrics
├── visualize_results.py      # Dashboard generation and thematic plotting
├── final_report.md           # Strategic breakdown of findings
├── historical_data.csv       # Source trade records (47MB)
└── fear_greed_index.csv      # Daily historical sentiment scores
```

---

## 🛠️ Quick Start

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**2. Run full analysis pipeline**
```bash
python analyze_and_train.py
```

**3. Generate dashboards & charts**
```bash
python visualize_results.py
```

---

## 📈 Performance Dashboard

> Run `python visualize_results.py` to generate `performance_dashboard.png` — a full analysis dashboard showcasing PnL by sentiment, strategy heatmap, and win rate trends.

---
*Developed for the Primetrade.ai AI Intern Assessment.*

**`BTC-SENT // V1.1.0`**