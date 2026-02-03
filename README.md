# ml_tradespace
Function to train ML models on a given tradespace dataframe, with a robust preprocessing pipeline.
# ML Tradespace Function Repository

This repository contains the **`ml_tradespace`** function, used to train, evaluate, and predict using multiple machine learning models on a tradespace dataset. This code was developed for thesis research on [Your Topic Here] and is designed for reproducibility and extensibility.

---

## Overview

The `ml_tradespace` function provides a **robust ML training pipeline** with:

- Flexible handling of **categorical and numerical features**
- Choice of **scaler for numeric columns** (`Min-Max` or `Standard`)
- Training and evaluation of **multiple ML models** per target
- Generation of predictions on new input data
- Extraction of **feature importances** for interpretability
- Timing measurements for training, testing, and predictions

The function supports both **regression and supervised learning models** from `scikit-learn`, `XGBoost`, `LightGBM`, and `Keras`.

---

## Repository Structure

ml_tradespace_repo/
│
├── README.md
├── LICENSE
├── requirements.txt # Python dependencies
├── src/
│ └── utils_v_final.py # Contains the core ml_tradespace function
├── examples/
│ └── example_usage.ipynb # Jupyter notebook demonstrating usage
├── data/
│ └── README.md # Instructions for obtaining input datasets

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ml_tradespace_repo.git
cd ml_tradespace_repo
pip install -r requirements.txt


**## Example Usage**

from src.ml_tradespace import ml_tradespace
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load your dataset
df = pd.read_csv("data/tradespace_data.csv")
input_df = pd.read_csv("data/new_inputs.csv")

# Define features, targets, and models
feats = ["feature_1", "feature_2", "feature_3"]
targets = ["target_1", "target_2"]
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=123),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=123)
}

# Run the ML tradespace function
results, trained_models, preds, df_importances, timing_info, df_test_results, preproc = ml_tradespace(
    df=df,
    feats=feats,
    targets=targets,
    train=0.8,
    inputs=input_df,
    models=models,
    encoder='One-Hot',
    scaler='Standard'
)

# View RMSE and R²
print(results)

# View predictions on new inputs
print(preds.head())

# View feature importances
print(df_importances.head())

LoCricchio, A. (2026). ML Tradespace Function. GitHub repository.
https://github.com/aloc5/ml_tradespace

