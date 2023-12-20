# Debt Recovery ML Solution

## Overview
This project involves developing a Machine Learning (ML) solution to enhance debt recovery processes. The system integrates ML models into debt management strategies to optimize recovery efforts and provide actionable insights.

## Structure
- **Database Setup**: Scripts to create and populate a SQLite database with debt-related data.
- **Data Processing**: Python code for data cleaning, preprocessing, and feature engineering.
- **Model Training and Evaluation**: Implementation of ML models including RandomForest, XGBoost, and Neural Networks for predictive analytics.

## Key Components
1. **SQLite Database**: Stores data related to debtors, debts, payments, communication logs, and legal actions.
2. **Data Preprocessing**: Cleans and prepares data for ML modeling.
3. **ModelWrapper Class**: Encapsulates the ML pipeline, including model training, evaluation, and serialization.

## How to Use
1. **Database Initialization**: Run the SQLite script to create and populate the database.
2. **Save Data File**: Execute the Python df file to save the sample dataset
3. **Run the Python Script**: Execute the Python model script to train and evaluate the models.

## Models Used
- **RandomForestClassifier**: For baseline predictive modeling.
- **XGBoostClassifier**: Advanced model for high-performance predictions.
- **Neural Network**: Implemented using TensorFlow for complex pattern recognition.

## Requirements
- Python 3.8+
- Libraries: pandas, sqlite3, sklearn, xgboost, tensorflow, joblib

## Setup and Installation
Ensure you have Python installed and then install the required packages:
```bash
pip install pandas sqlite3 scikit-learn xgboost tensorflow joblib
