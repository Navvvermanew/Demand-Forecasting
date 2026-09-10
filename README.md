# 📈 AI-Driven Demand Forecasting using Ensemble Time-Series Models

> A machine learning-based demand forecasting project that combines multiple regression models using a neural meta-learner to predict future seasonal sales.

---

## 📌 Overview

This project focuses on forecasting future seasonal sales for a fan company using advanced time-series analysis and machine learning techniques.

The dataset contains **1,039 time series**, with **38 data points per series**. Extensive exploratory data analysis was performed to identify important temporal patterns, seasonality, and correlations within the data.

Multiple machine learning regression models were trained and evaluated, including **Decision Tree, K-Nearest Neighbors (KNN), Lasso, XGBoost, and Elastic Net**. The predictions from these base models were then combined using a **neural meta-learner** to create an ensemble forecasting system.

The final ensemble achieved a **MAPE of 35.1%**, demonstrating its ability to handle short and highly seasonal sales histories.

---

## 🎯 Problem Statement

Accurate demand forecasting is important for effective inventory planning, production scheduling, and business decision-making.

The objective of this project is to:

- Analyze historical sales patterns
- Identify seasonality and temporal dependencies
- Engineer relevant forecasting features
- Train multiple machine learning regression models
- Combine model predictions using an ensemble approach
- Forecast future seasonal demand
- Evaluate forecasting performance using appropriate metrics

---

## 📊 Dataset

The dataset consists of:

- **1,039 individual time series**
- **38 observations per time series**
- Historical seasonal sales information
- Short and highly seasonal demand patterns

Each time series represents historical demand observations that are used to forecast future sales.

> **Note:** The dataset is included/not included in this repository depending on its redistribution permissions.

---

## 🔍 Exploratory Data Analysis

Comprehensive exploratory data analysis was performed to understand the underlying characteristics of the time-series data.

The analysis focused on:

- Time-series trends
- Seasonal patterns
- Demand variability
- Correlation analysis
- Distribution of observations
- Series-level characteristics
- Identification of important forecasting patterns

Example visualizations can be found in the `images/` directory.

---

## ⚙️ Methodology

The overall forecasting pipeline follows these steps:

```text
Historical Sales Data
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Multiple Base ML Models
        ↓
Model Predictions
        ↓
Neural Meta-Learner
        ↓
Ensemble Forecast
        ↓
Performance Evaluation
        ↓
Future Demand Prediction
