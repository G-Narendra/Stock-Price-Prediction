# 📈 Stock Price Prediction

**Time-series forecasting of stock prices using LSTM and traditional ML models.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

Time-series forecasting of stock prices using LSTM and traditional ML models.

---

## 📊 What I Built

Hybrid LSTM + Random Forest ensemble for stock price prediction with rolling window features.

### Key Results

| Metric | Value |
|---|---|
| **Model** | LSTM + Random Forest |
| **Train Size** | 70% |
| **Test Size** | 30% |
| **Evaluation** | r2_score, rmse |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas |
| **ML Framework** | Scikit-Learn |
| **Model** | LSTM + Random Forest |

---

## 📁 Project Structure

```
Stock-Price-Prediction/
├── *.ipynb                          # Main notebook with full pipeline
├── ml_evaluation_utils.py           # Evaluation utilities (CV, confidence intervals)
├── README.md
└── LICENSE
```

---

## 🔧 How to Run

```bash
# Install dependencies
pip install pandas scikit-learn jupyter

# Run the notebook
jupyter notebook *.ipynb
```

---

## 🧪 Engineering Decisions

| Decision | Rationale |
|---|---|
| **LSTM + Random Forest** | Chosen as baseline model for this problem type |
| **70/30 Split** | Standard split ratio for small-medium datasets |
| **Random State 2529** | Fixed random state ensures reproducibility |

---

## ⚠️ Limitations

- **No walk-forward validation**
- **Look-ahead bias risk**
- **No confidence intervals**

---

## ⚠️ Disclaimer

This is an educational project for learning ML concepts. It is not intended for production use.

---

*Built as part of MSc Data Science coursework — demonstrating fundamental ML pipeline.*
