# Multi-Agent CTR Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangGraph](https://img.shields.io/badge/LangGraph-AgenticAI-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## Project Overview

This project is an end-to-end Multi-Agent Machine Learning pipeline built using LangGraph to predict whether a user will click on an online advertisement (Click Through Rate Prediction).

Unlike traditional machine learning pipelines, this project uses an agent-based architecture where multiple independent agents handle specific responsibilities including data ingestion, feature engineering, behavioral scoring, model training, model evaluation, and automated business-aware final model selection.

The project demonstrates workflow orchestration, modular machine learning design, production-oriented model persistence, and real-time API deployment.

---

## Problem Statement

Predict whether a user will click on an advertisement based on demographic information, browsing behavior, and user engagement patterns.

Target Variable:

* clicked = 1 → User clicked advertisement
* clicked = 0 → User did not click advertisement

---

## Project Architecture

```text
Raw Dataset (CSV)
        │
        ▼
Data Agent
        │
        ▼
Feature Engineering Agent
        │
        ▼
User Behavior Agent
        │
        ▼
Prediction Agent
    ├── Logistic Regression
    ├── XGBoost
    └── Metrics Generation
        │
        ▼
Evaluation Agent
    ├── Precision-based Comparison
    ├── Business Logic Selection
    └── Save best_model.pkl
        │
        ▼
FastAPI Layer
    ├── Receive JSON Input
    ├── Internal Feature Encoding
    ├── Behavior Score Generation
    ├── Load Saved Model
    └── Return Prediction
```

---

## Agents Used

### Data Agent

* Loads dataset
* Checks missing values
* Validates dataset structure

### Feature Engineering Agent

* Encodes categorical variables using one-hot encoding
* Converts categorical data into machine learning compatible format

### User Behavior Agent

* Creates custom behavior_score feature
* Combines page views, session duration, and previous clicks for engagement analysis

### Prediction Agent

* Splits dataset into train-test sets
* Trains Logistic Regression model
* Trains XGBoost model
* Calculates Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

### Evaluation Agent

* Compares multiple model metrics
* Uses business-aware evaluation logic
* Prioritizes Precision over Accuracy
* Saves best performing model using Joblib

---

## Tech Stack

* Python
* Pandas
* Scikit-Learn
* XGBoost
* LangGraph
* FastAPI
* Joblib
* Git
* GitHub
* Machine Learning
* Classification Metrics

---

## Project Structure

```text
CTR_Prediction_System/

├── agents/
│   ├── data_agent.py
│   ├── feature_agent.py
│   ├── user_behavior_agent.py
│   ├── prediction_agent.py
│   └── evaluation_agent.py
│
├── pipeline/
│   ├── graph_builder.py
│   └── state.py
│
├── api/
│   └── main.py
│
├── models/
│   └── best_model.pkl
│
├── data/
│   └── ctr_dataset.csv
│
└── README.md
```

---

## Business Logic Used

For CTR Prediction systems, False Positives increase advertisement expenditure.

Therefore Precision was prioritized over Accuracy while selecting the final model.

Higher Precision = Lower False Positives = Better Advertisement Spending Efficiency.

---

## API Deployment

The project includes a FastAPI deployment layer for real-time CTR prediction.

### API Endpoint

POST /predict

### Sample Input

```json
{
  "user_id": 1,
  "age": 25,
  "previous_clicks": 8,
  "page_views": 15,
  "session_time": 300,
  "gender": "Female",
  "device_type": "Mobile",
  "ad_category": "Electronics",
  "time_of_day": "Morning"
}
```

### API Workflow

User JSON Input
↓
Internal Feature Encoding
↓
Behavior Score Generation
↓
Load Saved Model (best_model.pkl)
↓
Prediction
↓
JSON Response

### Sample Output

```json
{
  "prediction": 0
}
```

Prediction Meaning:

* 0 → User is unlikely to click advertisement
* 1 → User is likely to click advertisement

---

## Key Learnings

Through this project, I gained hands-on experience in:

* Multi-Agent workflow orchestration using LangGraph
* Machine Learning model comparison and evaluation
* Business-driven model selection beyond accuracy metrics
* Feature engineering and behavioral score creation
* Model persistence using Joblib
* REST API deployment using FastAPI
* Real-time machine learning inference workflows
* Git version control and GitHub project management

---

## Future Improvements

* Docker containerization
* Cloud deployment pipeline
* Real-time inference system
* Database integration for prediction logging

---

## Author

Saumya 
