# Attention-Based Time Series Forecasting

## Overview
This project implements a Transformer-based model for multivariate time series forecasting and compares it with an LSTM baseline.

## Features
- Synthetic correlated dataset
- Seasonality + non-stationarity
- Missing data handling
- Transformer with interpretable attention
- Baseline comparison

## Run Order
1. `train.py` – trains models
2. `evaluate.py` – prints MAE & RMSE
3. `attention_analysis.py` – visualizes attention

## Models
- Transformer Encoder with 3 layers, 4 heads
- LSTM baseline (64 hidden units)

## Metrics
MAE and RMSE are used to compare performance.
<img width="530" height="457" alt="image" src="https://github.com/user-attachments/assets/0df12d9d-a001-4635-b99f-b9e45252219a" />
