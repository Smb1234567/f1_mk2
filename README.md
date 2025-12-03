# F1 Race Outcome Predictor

A machine learning application to predict Formula 1 race outcomes using historical data.

## Overview
This project uses historical F1 race data to predict finishing positions based on:
- Driver performance history
- Team performance history  
- Circuit characteristics
- Grid position
- Race conditions

## Features
- Past Race Analysis: Compare model predictions vs actual results
- Future Race Prediction: Predict outcomes for upcoming races
- Multiple ML models: Random Forest, XGBoost, Linear Regression
- Ensemble predictions for improved accuracy

## Requirements
- Python 3.8+
- FastF1 library
- Streamlit
- Pandas, NumPy, Scikit-learn
- Plotly

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Run the app: `streamlit run app.py`

## Model Performance
- Mean Absolute Error: ~1.69 positions
- R² Score: ~0.846
- Uses ensemble of 3 models with weighted combination

## How It Works
The application:
1. Fetches historical race data from FastF1
2. Extracts relevant features (driver stats, team stats, circuit stats, etc.)
3. Trains machine learning models to predict race outcomes
4. Provides a Streamlit interface to analyze predictions vs actual results
5. Allows prediction of future races with configurable starting grid

## Data Sources
- F1 timing data via FastF1 library
- Historical race results from 2022-2024 seasons
- Driver and team performance metrics

## Repository Structure
- `app.py`: Streamlit web application
- `train_model.py`: Training pipeline
- `data_loader.py`: Data fetching and preprocessing
- `feature_builder.py`: Feature engineering
- `validate_and_retry.py`: Data validation and retry logic
- `data_tracker.py`: Data loading status tracking
- `config.py`: Configuration and constants
- `f1_model_artifacts.pkl`: Trained model (after training)
- `training_data_raw.csv`: Raw training data (after fetching)