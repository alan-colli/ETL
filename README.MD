# 📈 S&P 500 ETL & Market Analysis

ETL pipeline analyzing 50+ years of S&P 500 data to answer one question:
**How do market sectors behave during historical crises?**

---

## 🛠️ Tech Stack
`Python` → `PostgreSQL` → `Power BI`

---

## ⚙️ Pipeline
`Raw CSV` → `Extract` → `Clean` → `Transform` → `Load PostgreSQL`

**Metrics created:** `cumulative_return` · `drawdown` · `price_vs_sma200` · `sharpe_proxy` · `crisis_period`

---

## 💡 Key Insights
- Consumer Cyclical delivered the highest cumulative return since 1970
- Technology had the deepest drawdown during the Dot-com Bubble
- Utilities shows the best Sharpe Proxy — most stable risk/return ratio
- Consumer Defensive was the most resilient sector across all crises

---

## 🖥️ Dashboard Preview
![Market Overview](https://raw.githubusercontent.com/alan-colli/ETL/main/screenshots/MarketOverview.png)
![Historical Crises](https://raw.githubusercontent.com/alan-colli/ETL/main/screenshots/HistoricalCrisis.png)
![Risk vs Return](https://raw.githubusercontent.com/alan-colli/ETL/main/screenshots/RiskxReturn.png)
![Covid Crisis](https://raw.githubusercontent.com/alan-colli/ETL/main/screenshots/CovidCrisis.png)

---

## 🗃️ Dataset
1,014,000 rows · 98 companies · 11 sectors · 1970–2026

[Kaggle Source](https://www.kaggle.com/datasets/samyakrajbayar/s-and-p-500-complete-historical-dataset-50-years/data) — place CSV in `data/raw_data/`

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
# configure .env with PostgreSQL credentials
python scripts/pipeline.py
```
