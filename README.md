# 🌪️ Storm Event & Property Damage Predictor

## Overview
This project is an end-to-end machine learning application that predicts the most likely type of severe weather event (e.g., Tornado, Thunderstorm Wind, Hail) and estimates the associated property damage based on location and date. 

The back-end models are trained on historical NOAA storm data, and the front-end is an interactive web application built with Streamlit.

## Features
* **Storm Type Classification:** Uses a Random Forest Classifier to predict the event type based on Latitude, Longitude, and Date.
* **Damage Estimation:** Uses a Random Forest Regressor combined with One-Hot Encoded event categories to predict financial property damage.
* **Interactive Dashboard:** A Streamlit front-end that allows users to input coordinates and dates via sliders and instantly receive model predictions.
* **Data Visualizations:** Includes exploratory data analysis (EDA) and model accuracy metrics (like confusion matrices) using Seaborn and Matplotlib.

## Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `RandomForestRegressor`)
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit

## 📂 Project Structure
* `back.py`: The data cleaning, feature engineering, and Random Forest training pipeline.
* `front.py`: The interactive Streamlit web application.
* `storm_model.pkl`: The saved classification model for predicting storm types.
* `damage_model.pkl`: The saved regression model for estimating property damage.

## How to Run Locally
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn streamlit matplotlib seaborn
3. Run back.py
4. Run front.py and then enter: python -m streamlit run front.py, in the terminal