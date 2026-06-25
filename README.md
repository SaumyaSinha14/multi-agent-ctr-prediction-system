# Multi-Agent CTR Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangGraph](https://img.shields.io/badge/LangGraph-AgenticAI-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## Project Overview

An end-to-end Multi-Agent Machine Learning system built using LangGraph to predict whether a user will click on an online advertisement (Click Through Rate Prediction).

The system uses specialized agents for data ingestion, feature engineering, behavioral scoring, model training, evaluation, and automated business-aware model selection. The final model is deployed using FastAPI for real-time prediction serving.

---

## Problem Statement

Predict whether a user will click on an advertisement based on demographic information, browsing behavior, and engagement patterns.

Target Variable:

* clicked = 1 → User clicked advertisement
* clicked = 0 → User did not click advertisement

---

## System Architecture

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
    ├── Precision-based Business Logic
    ├── Final Model Selection
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

## Tech Stack

* Python
* Pandas
* Scikit-Learn
* XGBoost
* LangGraph
* FastAPI
* Joblib
* Git & GitHub
* Machine Learning
* Classification Metrics

---

## Business Logic

For CTR prediction systems, **False Positives directly increase advertisement expenditure**.

Instead of selecting models purely on Accuracy, the evaluation system prioritizes **Precision** to minimize unnecessary ad spending.

Higher Precision = Lower False Positives = Better Advertisement Efficiency.

---

## API Deployment

The project includes a FastAPI deployment layer for real-time prediction.

### Endpoint

```text
POST /predict
```

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

## Future Improvements

* Docker containerization
* Cloud deployment pipeline
* Database integration for prediction logging
* Real-time inference system

---

## Author

**Saumya**

