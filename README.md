# Transaction-Fraud-Detection-System-

# AI-Powered Transaction Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-99.94%25-brightgreen.svg)](.)

> Machine learning system achieving **99.94% ROC-AUC** on 6.3M+ transactions

## 📊 Project Highlights

**Best Model:** CatBoost  
**Dataset:** PaySim - 6,362,620 mobile money transactions  
**Challenge:** Highly imbalanced (0.13% fraud rate - 1:774 ratio)

### Performance Metrics

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **99.94%** |
| **Precision** | **98.23%** |
| **Recall** | **95.47%** |
| **F1-Score** | **96.83%** |

## 🎯 What I Built

An end-to-end fraud detection system that:
-  Analyzes 6.3M+ transactions in real-time
-  Detects 95%+ of fraudulent transactions
-  Minimizes false alarms (98% precision)
-  Provides explainable predictions using SHAP
-  Production-ready with sub-100ms inference

## 🔧 Technical Implementation

### Feature Engineering
Created **25+ predictive features** including:
- Amount transformations (log, sqrt, percentiles)
- Balance verification & mismatch detection
- Temporal patterns (hour, day analysis)
- Transaction type encodings
- Merchant identification indicators
- Behavioral risk scores

### Machine Learning Pipeline
- **Models:** XGBoost, LightGBM, CatBoost
- **Class Balancing:** SMOTE (handled 1:774 imbalance)
- **Explainability:** SHAP integration
- **Validation:** Stratified train-test split

### Tech Stack
`Python` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `SHAP` `Pandas` `NumPy` `Plotly` `SMOTE`

## 📈 Results Comparison

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|
