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

A high-performance data science pipeline correlating market sentiment signals with real-world trader outcomes — uncovering the emotional patterns that drive market alpha.

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

## 🔍 Market Insights

### 01 — Sell the Peak
> **Signal: Extreme Greed → SHORT**

Traders achieved the highest average PnL by taking short positions during **Extreme Greed** phases. When the crowd is most euphoric, the contrarian edge is at its sharpest — patience and positioning outperform momentum chasing.

### 02 — Sentiment Correlation
> **Signal: High Greed → High Variance**

Regression analysis shows a clear broadening of PnL distribution as sentiment values increase — indicating higher volatility and higher potential returns during Greed cycles. Risk and reward scale with euphoria.

### 03 — Fear Accumulation Edge
> **Signal: Fear Zone 25–45 → BUY**

Buying during Fear (25–45) significantly outperformed Extreme Fear zones. Extreme Fear often precedes continued downtrends — not capitulation bottoms. Discriminating between them is key to avoiding falling knives.

---

## 🧠 ML Model — Intelligence Layer

### Model
```
HistGradientBoostingClassifier
Upgraded from Random Forest — high-density ensemble with dynamic feature engineering
```

### Training Results

| Parameter | Value |
|:---|:---|
| **Model** | `HistGradientBoostingClassifier` |
| **Final Accuracy** | **95.80%** |
| **Split** | 80% Train / 20% Test |
| **Strategy** | Shuffled historical data |
| **Target** | Trade profitability (binary) |

```
ACCURACY  ████████████████████████████████████████████████░░  95.8%
```

### Feature Engineering Matrix

| Feature Category | Description | Impact |
|:---|:---|:---:|
| **Price Relativity** | Execution price vs. daily average for each specific coin — identifies "value" entries | 🔴 CRITICAL |
| **Sentiment Momentum** | MA3 moving averages and daily delta of Fear & Greed values — captures shifting emotion | 🟡 HIGH |
| **Categorical Meta** | Account IDs and asset encodings — adjusts for individual trader alpha levels | 🟡 HIGH |
| **Time-Based** | Hour of day and day of week execution patterns — uncovers liquidity windows | 🔵 MED |

---

## 📁 Project Architecture

```
btc-sent/
├── analyze_and_train.py      # Master pipeline — ingestion, cleaning, feature engineering & model training
├── visualize_results.py      # Generates dark-themed dashboards and all output figures
├── final_report.md           # Strategic breakdown of trade behaviors and recommended sentiment strategies
├── historical_data.csv       # Source trade records from Hyperliquid — 47MB, 211k+ rows
└── fear_greed_index.csv      # Daily historical sentiment scores (0–100)
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

**4. Review strategy report**
```bash
open final_report.md
```

---

## 📈 Performance Dashboard

> Run `python visualize_results.py` to generate `performance_dashboard.png` — a full analysis dashboard showcasing PnL by sentiment, strategy heatmap, and win rate trends.

---

*Developed for the Primetrade.ai AI Intern Assessment.*

**`BTC-SENT // V1.1.0`**