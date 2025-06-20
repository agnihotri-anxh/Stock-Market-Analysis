# Cryptocurrency Price Prediction Dashboard

A comprehensive cryptocurrency price prediction dashboard using multiple machine learning models.

## Features

- Real-time data fetching from Yahoo Finance
- Interactive visualizations using Plotly
- 10 different prediction models:
  1. Prophet
  2. LSTM (Deep Learning)
  3. ARIMA
  4. SARIMA
  5. XGBoost
  6. LightGBM
  7. CatBoost
  8. Random Forest
  9. SVR (Support Vector Regression)
  10. Ensemble (Coming soon)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Using the Dashboard

1. The dashboard will automatically load with the latest BTC-USD data
2. Use the sidebar to select different prediction models
3. View the historical data visualization
4. Check model predictions and performance metrics

## Model Details

- **Prophet**: Facebook's time series forecasting model
- **LSTM**: Deep Learning model for sequence prediction
- **ARIMA**: Statistical method for time series forecasting
- **SARIMA**: Seasonal ARIMA model
- **XGBoost**: Gradient boosting model
- **LightGBM**: Light Gradient Boosting Machine
- **CatBoost**: Gradient boosting on decision trees
- **Random Forest**: Ensemble learning method
- **SVR**: Support Vector Regression
- **Ensemble**: Combination of multiple models (Coming soon)

## Performance Metrics

The dashboard displays three key metrics for each model:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## Data Features

The dashboard uses the following features for prediction:
- Closing Price
- Returns
- Moving Averages (7, 30, and 90 days)
- Volatility 