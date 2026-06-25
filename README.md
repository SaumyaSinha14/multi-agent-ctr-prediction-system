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

Data Agent
↓
Feature Agent
↓
User Behavior Agent
↓
Prediction Agent
├── Logistic Regression
├── XGBoost
└── Model Metrics Generation
↓
Evaluation Agent
├── Business-Aware Model Comparison
├── Final Model Selection
└── Model Saving using Joblib
↓
END

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

## Future Improvements

* FastAPI integration for prediction API
* Docker containerization
* Deployment pipeline
* Real-time inference system

---

## Author

Saumya Sinha
