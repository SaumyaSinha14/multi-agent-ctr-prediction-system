# Multi-Agent CTR Prediction System

## Project Overview

This project is an end-to-end Multi-Agent Machine Learning pipeline built using LangGraph to predict whether a user will click on an advertisement (Click Through Rate Prediction).

Instead of using a traditional monolithic machine learning workflow, the system follows an agent-based architecture where each agent performs an independent responsibility such as data ingestion, feature engineering, behavioral scoring, model training, evaluation, and final model selection.

The project demonstrates modular AI workflow orchestration, business-aware model evaluation, and production-oriented model persistence.

---

## Problem Statement

Predict whether a user will click an online advertisement based on demographic information, browsing behavior, and engagement patterns.

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
    ├── Load Saved Model
    └── Return Prediction
```
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

## Agents Used

### Data Agent

* Loads dataset
* Checks missing values
* Validates dataset structure

### Feature Agent

* Encodes categorical variables using one-hot encoding
* Converts categorical data into machine learning friendly format

### User Behavior Agent

* Creates custom behavior_score feature using user engagement patterns
* Combines page views, previous clicks, and session duration

### Prediction Agent

* Splits dataset into training and testing data
* Trains Logistic Regression model
* Trains XGBoost model
* Calculates Accuracy, Precision, Recall, F1 Score, Confusion Matrix

### Evaluation Agent

* Compares multiple model metrics
* Uses business logic for final model selection
* Prioritizes Precision to reduce False Positives
* Saves final selected model using Joblib

---

## Tech Stack

Python
Pandas
Scikit-Learn
XGBoost
LangGraph
Joblib
Git
GitHub
Machine Learning
Classification Metrics

---

## Business Logic Used

For CTR Prediction systems, False Positives increase advertisement expenditure.

Therefore Precision was prioritized over Accuracy while selecting the final model.

Higher Precision = Lower False Positives = Better Advertisement Spending Efficiency

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

Prediction Meaning:

0 → User is unlikely to click advertisement
1 → User is likely to click advertisement
```
## Future Improvements

* Docker containerization
* Deployment pipeline
* Real-time inference system

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

## Author

Saumya
