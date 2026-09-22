# Retail Sales Forecasting using Ensemble Machine Learning and Neural Stacking

## 📌 Project Overview

This project develops a multi-series retail sales forecasting framework using historical sales, promotional activity, temporal features, lag variables, and rolling statistics.

The project uses the **Corporación Favorita Store Sales Forecasting** dataset and models sales across **1,782 complete store–product-family time series**.

Five machine-learning models were evaluated:

* Decision Tree
* K-Nearest Neighbors (KNN)
* LASSO Regression
* Elastic Net Regression
* XGBoost

A **Neural Stacking** model was subsequently developed using out-of-fold predictions from the five base learners as meta-features.

The complete pipeline follows a time-aware forecasting methodology to avoid random train-test leakage.

---

## 🎯 Objectives

* Forecast monthly retail sales across multiple store-product series.
* Capture temporal dependencies using lag and rolling-window features.
* Compare linear, distance-based, tree-based, and boosting models.
* Evaluate models using expanding-window time-series validation.
* Develop an out-of-fold neural stacking ensemble.
* Evaluate the final models on a completely unseen future test period.

---

## 📊 Dataset

**Dataset:** Corporación Favorita Grocery Sales Forecasting

The original `train.csv` contains:

* 3,000,888 observations
* 54 stores
* 33 product families
* Daily sales observations
* Promotion information

For this project, the daily data was aggregated to monthly frequency.

### Modeling Dataset

| Property                        |               Value |
| ------------------------------- | ------------------: |
| Complete store-family series    |               1,782 |
| Monthly observations per series |                  38 |
| Modeling period                 | Jan 2014 – Feb 2017 |
| Feature-engineered observations |              46,332 |
| Final development observations  |              35,640 |
| Final test observations         |              10,692 |

The final six months, **September 2016 – February 2017**, were completely reserved as an unseen test period.

---

## 🔬 Exploratory Data Analysis

The exploratory analysis identified:

* Strong heterogeneity across store-product series.
* An overall upward sales trend.
* Calendar-month seasonality.
* Large differences in sales scale between product families.
* A substantial number of zero-sales observations.

Average monthly sales across the modeling dataset were approximately **11,688 units**, with substantial variation between individual series.

---

## ⚙️ Feature Engineering

The forecasting model uses historical and temporal information available at prediction time.

### Lag Features

* `lag_1`
* `lag_2`
* `lag_3`
* `lag_6`
* `lag_12`

These capture short-term and annual temporal dependencies.

### Rolling Features

* `rolling_mean_3`
* `rolling_mean_6`
* `rolling_std_3`

All rolling statistics were calculated after shifting the target by one period to prevent future information from entering the features.

### Calendar Features

* `month_num`
* `year`
* `time_idx`

### Additional Features

* `store_nbr`
* `family`
* `onpromotion`

---

## 🧠 Modeling Pipeline

```text
Daily Retail Sales Data
          ↓
Monthly Aggregation
          ↓
Complete Store × Family Series
          ↓
Lag & Rolling Feature Engineering
          ↓
Time-Based Development / Test Split
          ↓
Expanding-Window Cross Validation
          ↓
┌─────────────────────────────────────┐
│          Base Learners              │
│                                     │
│ Decision Tree                       │
│ KNN                                 │
│ LASSO                               │
│ Elastic Net                         │
│ XGBoost                             │
└─────────────────────────────────────┘
          ↓
Out-of-Fold Predictions
          ↓
Neural Meta-Learner
          ↓
Final Unseen Test Evaluation
```

---

## ⏱️ Time-Series Validation Strategy

Random train-test splitting was avoided because it can introduce temporal leakage in forecasting problems.

Instead, an expanding-window validation strategy was used.

| Fold | Training Period     | Validation Period   |
| ---- | ------------------- | ------------------- |
| 1    | Jan 2015 – Dec 2015 | Jan 2016 – Feb 2016 |
| 2    | Jan 2015 – Feb 2016 | Mar 2016 – Apr 2016 |
| 3    | Jan 2015 – Apr 2016 | May 2016 – Jun 2016 |
| 4    | Jan 2015 – Jun 2016 | Jul 2016 – Aug 2016 |

This generated:

**14,256 out-of-fold predictions**

for training the neural stacking model.

---

## 🤖 Base Models

### 1. Decision Tree

```text
max_depth = 10
min_samples_leaf = 10
```

### 2. KNN

```text
n_neighbors = 10
weights = distance
p = 2
```

### 3. LASSO

```text
alpha = 1.0
max_iter = 100000
```

### 4. Elastic Net

```text
alpha = 1.0
l1_ratio = 0.5
max_iter = 100000
```

### 5. XGBoost

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

---

## 🧩 Neural Stacking

Instead of training the meta-learner on in-sample predictions, predictions generated through the expanding-window validation folds were used.

The five base-model predictions became the input features:

```text
Decision Tree Prediction
KNN Prediction
LASSO Prediction
Elastic Net Prediction
XGBoost Prediction
             ↓
       Neural Network
             ↓
       Final Forecast
```

The neural meta-learner used a compact multilayer perceptron:

```text
Input: 5 base-model predictions

Dense Layer: 32 neurons
        ↓
Dense Layer: 16 neurons
        ↓
Output: Sales Forecast
```

Early stopping was used to reduce overfitting.

---

## 📏 Evaluation Metrics

Four metrics were used:

### MAE

Mean Absolute Error measures the average absolute forecasting error.

### RMSE

Root Mean Squared Error penalizes large forecasting errors more heavily than MAE.

### MAPE

Mean Absolute Percentage Error was calculated only for observations with positive actual sales because the dataset contains zero-sales observations.

### sMAPE

Symmetric Mean Absolute Percentage Error was included as an additional percentage-based metric.

Because this dataset contains many small and zero-sales observations, percentage metrics can become extremely large. Therefore, MAE and RMSE are also reported prominently.

---

# 📈 Final Test Results

The final test period was:

**September 2016 – February 2017**

containing **10,692 observations**.

| Model           |          MAE |         RMSE |      MAPE |   sMAPE |
| --------------- | -----------: | -----------: | --------: | ------: |
| Decision Tree   |     1,860.96 |     7,088.61 |    82.89% |  47.12% |
| KNN             |     2,056.20 |     7,059.30 |    94.68% |  51.67% |
| LASSO           |     2,355.28 |     5,660.97 | 2,004.23% | 101.84% |
| Elastic Net     |     2,498.43 |     5,880.86 | 1,888.20% |  99.49% |
| **XGBoost**     | **1,486.79** | **5,537.97** |   100.09% |  66.39% |
| Neural Stacking |     2,603.35 |     6,558.59 | 2,148.45% |  94.41% |

### Key Result

XGBoost achieved the lowest final-test:

* **MAE: 1,486.79**
* **RMSE: 5,537.97**

The neural stacking model did not outperform the individual base learners on the final unseen test period.

This result is retained as part of the experiment rather than artificially selecting or modifying the methodology to force an ensemble improvement.

---

## 📊 Visualizations

### Model Comparison — MAE

![Model Comparison MAE](model_comparison_mae.png)

### Model Comparison — RMSE

![Model Comparison RMSE](model_comparison_rmse.png)

### Actual vs XGBoost Forecast

![Actual vs XGBoost Forecast](actual_vs_xgboost_forecast.png)

---

## 🔍 Key Findings

1. **XGBoost provided the strongest overall final-test performance in terms of MAE and RMSE.**

2. **Tree-based models handled the nonlinear relationships in the retail sales data effectively.**

3. **Linear models struggled with the highly heterogeneous and nonlinear sales distributions**, particularly when evaluated using percentage-based metrics.

4. **KNN provided competitive performance**, but its performance varied across validation periods.

5. **Neural stacking did not automatically improve forecasting accuracy.** The meta-learner performed worse than the strongest individual base model on the final test period.

6. **MAPE is highly sensitive to low-volume and zero-sales observations**, making MAE, RMSE, and sMAPE important complementary metrics.

7. The experiment demonstrates that **ensemble complexity does not necessarily translate into better out-of-sample forecasting performance**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Kaggle Dataset

---

## 📁 Suggested Repository Structure

```text
retail-sales-forecasting/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── retail_sales_forecasting.ipynb
│
├── visualizations/
│   ├── model_comparison_mae.png
│   ├── model_comparison_rmse.png
│   └── actual_vs_xgboost_forecast.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
jupyter
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the Corporación Favorita Store Sales Forecasting dataset from Kaggle and place `train.csv` in the appropriate data directory.

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/retail_sales_forecasting.ipynb
```

and execute the cells sequentially.

---

## 📌 Limitations

* The model uses monthly aggregation, which removes some daily-level demand information.
* MAPE is unstable for low-volume and zero-sales observations.
* The neural stacking model did not improve the final test performance.
* Hyperparameters were selected as baseline configurations rather than through an exhaustive optimization process.
* External factors such as holidays, oil prices, weather, macroeconomic indicators, and detailed store-level characteristics were not incorporated.

---

## 🔮 Future Improvements

Potential extensions include:

* Hyperparameter optimization using time-series cross-validation.
* LightGBM/CatBoost comparison.
* Hierarchical forecasting across store and product levels.
* Holiday and event features.
* More sophisticated promotion features.
* Log-transformed target modeling.
* Quantile forecasting and prediction intervals.
* Direct multi-step forecasting strategies.
* More advanced temporal architectures such as LSTM, Temporal Fusion Transformer, or N-BEATS.

---

## 👤 Author

**Naveen Verma**

IIT Kharagpur
Department of Mechanical Engineering

---

## ⭐ Project Summary

This project demonstrates an end-to-end **multi-series retail forecasting workflow**, combining time-series feature engineering, expanding-window validation, ensemble machine learning, out-of-fold prediction generation, and neural stacking.

The final evaluation uses a completely unseen future period, providing a realistic assessment of out-of-sample forecasting performance.
