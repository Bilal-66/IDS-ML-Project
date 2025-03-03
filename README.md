# NSL-KDD Intrusion Detection Project

This project is a machine learning pipeline for detecting network intrusions using the NSL-KDD dataset. The pipeline includes data preprocessing, model training, and a Flask API to serve predictions.

## Features

- Preprocess the NSL-KDD dataset by renaming columns and encoding categorical values.
- Train a Random Forest classifier on the preprocessed data.
- Evaluate model performance with metrics such as accuracy, confusion matrix, and classification report.
- Serve the trained model through a REST API built with Flask.

## Project Structure

- `src/preprocess.py`: Contains functions for loading, renaming, encoding, and splitting the dataset.
- `train.py`: Script to train the Random Forest model, evaluate its performance, and save the trained model.
- `src/app.py`: Flask application that loads the trained model and provides an endpoint for predictions.

## How to Run

1. **Set up the environment:**  
   Install required packages:
   ```bash
   pip install -r requirements.txt
