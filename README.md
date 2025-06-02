# 📈 Stock Market Prediction and Trend Analysis

This project implements a machine learning framework to predict the **S&P 500 index** using **70+ technical indicators** and state-of-the-art models including **SVR**, **XGBoost**, **Random Forest**, and **LSTM**. It is based on the research paper _“Key Technical Indicators for Stock Market Prediction”_.

## 📊 Objectives

- Predict future S&P 500 prices using technical indicators.
- Compare four models: SVR, XGBoost, Random Forest, and LSTM.
- Analyze the effect of lagged features (previous prices).
- Determine the most influential indicators using feature importance.

## 🧠 Models Used

| Model              | Description                                  |
|--------------------|----------------------------------------------|
| **SVR**            | Support Vector Regression with RBF kernel    |
| **XGBoost**        | Gradient Boosted Trees                       |
| **Random Forest**  | Ensemble of decision trees                   |
| **LSTM**           | Deep learning model for time series          |

Lagged and non-lagged versions of each model were evaluated.

## 📁 Project Structure  ├── code&report/
│ ├── model_new.py
│ ├── preprocessed_data.csv
│ ├── sp500_data.csv
│ └── final_report.pdf
├── research_paper/
│ └── file:///C:/Users/Dhruv/AppData/Local/Temp/079a9a7c-770c-4bc7-b77e-8bb8e57b29ef_1-s2.0-S2666827025000143-main%20(3).zip.9ef/1-s2.0-S2666827025000143-main%20(3).pdf
  
## 🔢 Technical Indicators

- **Trend**: SMA, EMA, WMA, TEMA, KAMA, HMA, etc.
- **Momentum**: RSI, MACD, ROC, KST, etc.
- **Volatility**: ATR, Bollinger Bands, Ulcer Index, etc.
- **Volume**: OBV, MFI, VWAP, CMF, etc.

Over 70 indicators were implemented using the `ta-lib` and fallback methods.

## ⚙️ Methodology

1. **Data Collection**: Historical data from 2000–2024 (`^GSPC` from Yahoo Finance)
2. **Feature Engineering**: Computation of technical indicators
3. **Preprocessing**: Min-Max Scaling and PCA (95% variance retained)
4. **Modeling**: Train/test split (80/20), training, and evaluation
5. **Evaluation**: MAE, MAPE, RMSE + visual plots

## 📈 Results

- **Best model**: `Random Forest with lag` (MAE = 0.0143)
- **Key features**: EMA, RSI, MACD, OBV, ATR
- **Lagged models** consistently outperformed non-lagged ones
- Feature importance and prediction plots included

## 📌 Key Findings

- Tree-based models (XGBoost, RF) outperformed LSTM and SVR
- Historical price trends (lagged data) are crucial
- Moving averages, volume, and momentum indicators are most predictive

## 🛠️ How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Run the analysis
python model_new.py

