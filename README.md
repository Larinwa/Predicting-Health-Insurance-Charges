# Predicting Health Insurance Charges Using a Neural Network (PyTorch)

## Overview
This project builds a fully connected neural network (Multilayer Perceptron) from scratch using PyTorch to predict individual medical insurance charges.  
It demonstrates data preprocessing, model design, training, and evaluation steps for a regression task using real-world structured data.

## Objectives
- Develop a regression model that predicts medical costs based on demographic and lifestyle features  
- Learn to implement and train a neural network using PyTorch  
- Evaluate the model’s performance using RMSE (Root Mean Squared Error)

## Data Preparation
- Dataset: `cleaned_insurance.csv`  
- Target Variable: `charges` (continuous medical costs)  
- Input Features:
  - age, bmi, children  
  - sex (one-hot encoded)  
  - smoker (one-hot encoded)  
  - region (one-hot encoded)
- Preprocessing:
  - Removed unnecessary index column  
  - Applied feature scaling using `StandardScaler`  
  - Split data into training and test sets (75% / 25%)

## Implementation Summary
- **Training Mode:** Model trained on 1003 samples  
- **Evaluation Mode:** Tested on 335 samples  
- **Performance Metrics:** Root Mean Squared Error (RMSE)

### Results
The model achieved reasonable predictive performance, showing consistent training convergence across epochs.

## Insights
- The neural network effectively captures nonlinear relationships between demographic features and medical costs.  
- Smoking status and BMI remain dominant predictors of high insurance charges.  
- Deeper architectures or regularization could further improve accuracy.

## Tools and Libraries
- Python  
- PyTorch  
- Pandas, NumPy  
- Scikit-learn (for preprocessing and metrics)
