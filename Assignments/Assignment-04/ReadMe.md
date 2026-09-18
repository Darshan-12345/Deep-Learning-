# Assignment 4: Time-Series Forecasting using LSTM Network

This directory contains the implementation of a Deep Learning model using Long Short-Term Memory (LSTM) networks for time-series forecasting.

---

## 📌 Problem Statement
> **Assignment Task:** Develop an LSTM-based model for time-series forecasting using stock price, weather, or sales datasets.

This project implements an end-to-end LSTM architecture for sequence prediction using a daily multivariate time-series dataset.

---

## 🛠️ Key Implementation Details

1. **Dataset:** Daily Time-Series Dataset (3,650 daily observations).
2. **Feature Engineering:**
   - Cyclical date encoding ($\sin$ and $\cos$ transformations on month indicators) to capture recurring seasonality.
3. **Data Preprocessing & Data Leakage Prevention:**
   - Independent scaling using `MinMaxScaler` fitted **strictly on training splits** for both features ($X$) and target ($y$).
   - Sequence generation using a 12-day lookback window (`time_steps = 12`).
   - Overlapped train-test tail indexing to preserve total test set sequence length.
4. **Model Architecture:**
   - **Input Layer:** Sequential shape `(time_steps, n_features)`
   - **LSTM Layer:** 64 units
   - **Dropout Layer:** 0.2 (to prevent overfitting)
   - **Dense Layer:** 32 units (`ReLU` activation)
   - **Output Layer:** 1 unit (Linear forecast)
5. **Optimization & Training:**
   - Loss Function: Mean Squared Error (`MSE`)
   - Optimizer: `Adam`
   - Callback: `EarlyStopping` (Patience = 15)

---

## 📊 Model Performance & Results

The model achieved strong convergence and seasonal forecasting performance:

* **Mean Absolute Error (MAE):** `1.71`
* **Root Mean Squared Error (RMSE):** `2.17`
* **$R^2$ Score:** `0.7191` (71.91% variance explained)

---

## 📁 Repository Structure

```text
Assignments/
└── Assignment-04/
    ├── assignment_NO_4_DL.ipynb    # Main Jupyter/Colab Notebook
    └── README.md                   # Project documentation
