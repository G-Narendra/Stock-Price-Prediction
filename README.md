# **📈 Stock Price Prediction using Machine Learning**

*A machine learning-based approach to predict Google's stock closing prices.*

## 🌟 **Overview**
This project aims to predict **Google's stock closing price for the next trading day** using historical stock prices. The dataset contains **daily stock records** with attributes such as **Open, High, Low, Close, Adjusted Close, and Volume.**

## 📊 **Dataset Overview**
- **Source**: GOOG.csv (Historical stock data of Google)
- **Records**: Multiple years of daily stock prices
- **Features**:
  - **Date**: Trading date
  - **Open, High, Low, Close**: Stock price values per day
  - **Adj Close**: Adjusted closing price
  - **Volume**: Number of shares traded
  - **Prev Close**: Previous day's closing price (engineered feature)

## 🎯 **Project Workflow**
✅ **Data Cleaning & Preprocessing** – Handling missing values, feature engineering.  
✅ **Feature Engineering** – Creating new relevant features like **Prev Close**.  
✅ **Model Training & Evaluation** – Comparing multiple regression models.  
✅ **Stock Price Visualization** – Plotting actual vs. predicted prices.  
✅ **Best Model Selection** – Identifying the most accurate model for forecasting.  

## 🛠️ **Tech Stack**
🔹 **Programming Language:** Python  
🔹 **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow (LSTM), Matplotlib, Seaborn  
🔹 **Model Type:** Regression (Linear Regression, Decision Tree, Random Forest, SVR, LSTM)  
🔹 **Development Environment:** Jupyter Notebook / Python Script  

## 📂 **Project Structure**
```
Stock-Price-Prediction/
├── stock_prediction.py       # Python script with model implementation
├── GOOG.csv                  # Dataset used for training/testing
├── stock_price_prediction.png # Visualizations of stock predictions
├── stocks_analysis_intro.txt  # Dataset overview
├── stocks_analysis_report.txt # Detailed project report
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
```

## 🚀 **Installation & Setup**
1️⃣ **Clone the Repository**  
```sh
git clone https://github.com/G-Narendra/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```
2️⃣ **Install Dependencies**  
```sh
pip install -r requirements.txt
```
3️⃣ **Run the Prediction Model**  
```sh
python stock_prediction.py
```

## 📈 **Model Performance & Evaluation**
Several regression models were trained and evaluated:

| Model                 | MSE  | R² Score |
|----------------------|------|----------|
| **Linear Regression** | 3.794 | 0.987 |
| **Decision Tree Regressor** | 9.421 | 0.964 |
| **Random Forest Regressor** | **3.466** | **0.988** |
| **Support Vector Regressor** | 16.420 | 0.934 |
| **LSTM Neural Network** | 6.104 | 0.978 |

### **Best Performing Model: Random Forest Regressor**
The **Random Forest Regressor** outperformed all other models with **the lowest MSE (3.466) and highest R² (0.988)**, making it the most reliable for predicting Google's next-day closing price.

## 🔍 **Visualization**
A comparison plot of **actual vs. predicted stock prices** was created to visually evaluate model accuracy.

## 📉 **Conclusion**
This project successfully developed and compared **multiple regression models** for stock price prediction. The **Random Forest Regressor** achieved the best accuracy. Future improvements could involve:
- **Adding more historical data for long-term forecasting**.
- **Incorporating additional technical indicators**.
- **Using deep learning models (LSTMs & Transformers) for improved predictions**.

## 🤝 **Contributions**
💡 Open to improvements! Feel free to:
1. Fork the repo  
2. Create a new branch (`feature-branch`)  
3. Make changes & submit a PR  


## 📩 **Connect with Me**
📧 **Email:** [narendragandikota2540@gmail.com](mailto:narendragandikota2540@gmail.com)  
🌐 **Portfolio:** [G-Narendra Portfolio](https://g-narendra-portfolio.vercel.app/)  
💼 **LinkedIn:** [G-Narendra](https://linkedin.com/in/g-narendra/)  
👨‍💻 **GitHub:** [G-Narendra](https://github.com/G-Narendra)  

⭐ **If you find this project useful, drop a star!** 🚀

