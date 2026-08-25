# 📈 Stock Price Prediction
### Time-Series Forecasting of Google (Alphabet Inc.) Stock Prices

<p align="center">
<img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Scikit--Learn-Regression-F7931E?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow">
<img src="https://img.shields.io/badge/R%C2%B2%20Score-0.988-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Dataset-GOOG%20Historical-4285F4?style=for-the-badge&logo=google">
</p>

---

## 🌟 Overview

Predicting stock market trends is a complex challenge characterized by non-linearity and volatility. This project implements a high-precision **Machine Learning pipeline** to forecast **Google’s (GOOG) closing prices**. By analyzing years of historical trading data, the system identifies patterns to assist in short-term financial decision-making.



---

## 📊 Dataset Overview

The study utilizes a decade-scale historical dataset of Alphabet Inc. (Class C) stock.

* **Primary Features:** Open, High, Low, Close, Adjusted Close, and Volume.
* **Engineered Feature:** `Prev Close` – Captures the immediate historical momentum by shifting the closing price by one day.
* **Target Variable:** `Close` – The specific price at the end of the next trading day.

---

## 🎯 Project Workflow

1.  **Data Preprocessing:** Handling outliers, scaling features, and managing time-series gaps.
2.  **Feature Engineering:** Creating lag variables (e.g., `Prev Close`) and rolling averages to capture trends.
3.  **Cross-Validation:** Using time-series splits to ensure models generalize across different market cycles.
4.  **Neural Network Integration:** Implementing **LSTM (Long Short-Term Memory)** to specifically capture long-term temporal dependencies.
5.  **Benchmarking:** Comparing traditional regression against ensemble methods and deep learning.



---

## 🧠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Machine Learning** | Scikit-learn |
| **Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📁 Project Structure

```bash
Stock-Price-Prediction/
├── src/
│   └── stock_prediction.py      # Regression & LSTM implementations
├── GOOG.csv                     # Historical Alphabet stock data
├── assets/
│   └── prediction_plot.png      # Actual vs. Predicted visual
├── requirements.txt             # Dependencies
└── README.md                    # Documentation

```


## ⚙️ Installation & Setup

### 1️⃣ Clone & Navigate

```bash
git clone [https://github.com/G-Narendra/Stock-Price-Prediction.git](https://github.com/G-Narendra/Stock-Price-Prediction.git)
cd Stock-Price-Prediction

```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt

```

### 3️⃣ Execute the Pipeline

```bash
python stock_prediction.py

```

---

## 📈 Model Performance & Evaluation

The following metrics represent the model's ability to fit the test data (Mean Squared Error & Coefficient of Determination):

| Model | MSE | R² Score |
| --- | --- | --- |
| **Random Forest Regressor** | **3.466** | **0.988** |
| **Linear Regression** | 3.794 | 0.987 |
| **LSTM Neural Network** | 6.104 | 0.978 |
| **Decision Tree Regressor** | 9.421 | 0.964 |
| **Support Vector Regressor** | 16.420 | 0.934 |

### **🏆 Champion Model: Random Forest Regressor**

The **Random Forest** ensemble outperformed deep learning in this specific dataset, achieving a near-perfect **0.988 R² Score**. Its ability to minimize variance through tree bagging makes it highly resilient to individual "noisy" trading days.

---

## 🚀 Future Roadmap

* [ ] **Technical Indicators:** Adding RSI, MACD, and Bollinger Bands as input features.
* [ ] **Sentiment Fusion:** Integrating live financial news sentiment scores using NLP.
* [ ] **Hybrid Model:** Developing a gated ensemble of LSTM and Random Forest for enhanced stability.

---

## Engineering Decisions & Challenges Solved

| Challenge | Decision | Why |
|---|---|---|
| Stock prices are non-stationary (trends change) | Rolling window normalization + lag features | Raw prices aren't stationary — rolling statistics capture local patterns the model can learn |
| LSTM alone underperforms on noisy data | Hybrid ensemble: LSTM for temporal patterns + Random Forest for feature-based predictions | LSTM captures sequential dependencies; RF captures non-linear feature interactions — combining them improves robustness |
| Future data leakage in time series | Strict temporal train/test split — no random shuffling | Random splits leak future information into training — temporal splits simulate real deployment conditions |
| Single-stock results don't generalize | Multiple ticker evaluation with per-stock metrics | A model that works on Apple may fail on Tesla — evaluating across stocks tests generalization |

## 👨‍💻 Author

**Narendra (G‑Narendra)** AI | ML | Python | Full Stack | GenAI Enthusiast

📧 [Email Me](mailto:narendragandikota2540@gmail.com) | 💼 [LinkedIn](https://linkedin.com/in/g-narendra/) | 👨‍💻 [GitHub](https://github.com/G-Narendra)

---

<p align="center">⭐ If you find this project useful, feel free to give it a star! 🚀</p>

