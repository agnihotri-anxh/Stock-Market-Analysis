# Zidio_Work

## Time Series Analysis and Stock Market Forecasting

This project is a web-based application for time series analysis and stock market forecasting using a variety of machine learning and statistical models. It allows users to input a stock symbol, select a forecasting model, and view detailed visualizations of historical data, model predictions, and future forecasts.

## Features

- **Web-Based Interface**: A user-friendly web UI built with Flask and Bootstrap.
- **Multiple Model Support**: Implements a wide range of forecasting models, including:
  - Statistical models (ARIMA, SARIMA)
  - Machine Learning models (XGBoost, LightGBM, Random Forest, SVR, Ridge)
  - Deep Learning models (LSTM)
  - Facebook's Prophet model
- **Comprehensive Visualization**: For each model, the application generates and displays:
  - Closing price history
  - Moving averages (100 and 200 days)
  - Actual vs. Predicted values on a test set
  - Future forecast for the next 30 days
- **Model Evaluation**: Calculates and displays the Mean Squared Error (MSE) for test set predictions to help evaluate model performance.
- **Batch Plot Generation**: Includes a script to generate all plots for all models at once for offline analysis and comparison.

## Screenshots

<img width="1838" height="900" alt="image" src="https://github.com/user-attachments/assets/6d6c41b1-2937-4fa8-b8f0-4211e0e62cba" />


## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## How to Use

### 1. Run the Web Application

To start the Flask web server, run the following command:

```sh
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000/`.

-   **Enter a stock symbol** (e.g., `POWERGRID.NS`, `AAPL`, `GOOGL`).
-   **Select a forecasting model** from the dropdown menu.
-   **Click "Predict"** to view the analysis and plots for the selected model.

### 2. Generate All Plots in Batch

To generate all 5 plots for every model at once, run the batch script:

```sh
python generate_all_model_plots.py
```

All generated plots will be saved in the `static/` directory, named according to the model and plot type (e.g., `static/LSTM_plot1_close_vs_time.png`).

## File Structure

```
.
├── app.py                      # Main Flask application
├── generate_all_model_plots.py # Script to generate all plots for all models
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # HTML template for the web UI
└── static/
    └── (generated plots)       # Directory where plot images are saved
```

## Models Supported

-   ARIMA
-   SARIMA
-   Prophet
-   LSTM (Long Short-Term Memory)
-   XGBoost
-   LightGBM
-   CatBoost
-   Random Forest
-   SVR (Support Vector Regression)
-   Ridge Regression

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please feel free to open an issue or submit a pull request. 
