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
