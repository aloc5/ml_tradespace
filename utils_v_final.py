#### Imports and Utilities ####
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.io as pio

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Importing relevant models
from sklearn.linear_model import (
    BayesianRidge, Ridge, RidgeCV, LinearRegression, LassoCV, Lars, LarsCV, LassoLarsCV, LassoLarsIC,
    ElasticNetCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, RANSACRegressor, HuberRegressor
)
from sklearn.base import clone
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import StackingRegressor
import itertools

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

import time

# other core imports you already have …

from keras import layers, models as km, callbacks, optimizers, losses
#                                   └────── aliased here ──────┘
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder

from sklearn.inspection import permutation_importance   # optional, slow

# utils_sampling.py  ──────────────────────────────────────────────────────────
from typing import Mapping, Sequence, Tuple, Any, Union, Optional
from math import ceil

import optuna
from sklearn.model_selection import cross_val_score


from tensorflow.keras import layers, models, optimizers, losses, callbacks


################################# BASIC ML TRADESPACE FUNCTION #########################################
def ml_tradespace(df, feats, targets, train, inputs, models, encoder, scaler):
    """
    Function to train ML models on a given tradespace dataframe, with a robust preprocessing pipeline.

    :param df: DataFrame containing the training and testing data.
    :param feats: List of feature column names.
    :param targets: List of target column names.
    :param train: The proportion of the dataset to use for training (e.g., 0.8).
    :param inputs: DataFrame of feature values for which to generate predictions.
    :param models: Dictionary of ML models, where keys are names and values are model objects.
    :param encoder: A string specifying the encoder. MUST be 'One-Hot'for this updated function.
    :param scaler: String specifying the scaler for numerical features ('Min-Max' or 'Standard').
    :return: A tuple of (results, trained_models, preds, df_importances, timing_info, df_test_results, preproc).
            results: dict of model performance metrics per target of testing data.
            trained_models: dict of trained model objects per target.
            preds: DataFrame of predictions on `inputs`.
            df_importances: DataFrame of feature importances for each model and target.
            timing_info: dict of timing information for each model.
            df_test_results: DataFrame of test results for each model and target.
            preproc: The fitted preprocessing pipeline.

    """
    ##########          1. Check Inputs (Integrated from Function 1)          ##########
    if df.empty:
        raise ValueError("The input DataFrame 'df' is empty.")
    if inputs.empty:
        raise ValueError("The prediction DataFrame 'inputs' is empty.")

    missing_feats = [feat for feat in feats if feat not in df.columns]
    if missing_feats:
        raise ValueError(f"The following features are missing from the DataFrame: {missing_feats}")
    
    missing_targets = [target for target in targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"The following targets are missing from the DataFrame: {missing_targets}")

    if not models or not isinstance(models, dict):
        raise ValueError("The 'models' parameter should be a non-empty dictionary of model objects.")

    if not (0 < train < 1):
        raise ValueError("The 'train' parameter must be a float between 0 and 1.")

    # Enforce the use of the robust ColumnTransformer pipeline
    if encoder != 'One-Hot':
        raise ValueError("This function now uses a robust ColumnTransformer pipeline. "
                         "Please set encoder='One-Hot'. The 'Label' encoder option has been removed "
                         "as it was methodologically flawed.")

    ##########          2. Split the data          ##########
    X = df[feats]
    y_list = [df[target] for target in targets]

    # Replace NaNs in categorical columns with the string "None"
    X = X.copy()
    X[feats] = X[feats].fillna("None")
    inputs = inputs.copy()
    inputs[feats] = inputs[feats].fillna("None")
    
    splits = train_test_split(X, *y_list, test_size=1.0 - train, random_state=123)
    X_train, X_test = splits[0], splits[1]
    y_train = dict(zip(targets, splits[2::2]))
    y_test  = dict(zip(targets, splits[3::2]))
    X_train = X_train.fillna("None")
    X_test = X_test.fillna("None")

    ##########          3. Create Preprocessing Pipeline (Integrated from Function 1)          ##########


    num_cols = X_train.select_dtypes(exclude='object').columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    if scaler == 'Min-Max':
        scaler_obj = MinMaxScaler()
    elif scaler == 'Standard':
        scaler_obj = StandardScaler()
    else:
        scaler_obj = 'passthrough' # No scaling on numeric columns

    # Build the preprocessing transformer using the best-practice ColumnTransformer
    transformers = []
    if num_cols:
        transformers.append(('num', scaler_obj, num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))

    preproc = ColumnTransformer(transformers, remainder='passthrough')

    # Fit-transform training data; transform test data
    X_train_scaled = preproc.fit_transform(X_train)
    X_test_scaled  = preproc.transform(X_test)

    ##########          4. Loop through and Train Models          ##########
    trained_models = {t: {} for t in targets}
    results        = {t: {} for t in targets}
    timing_info    = {t: {} for t in targets}
    test_set_preds = {}

    for t in targets:
        y_tr = y_train[t]
        y_te = y_test[t]
        test_set_preds[f"{t}_actual"] = y_te

        for model_name, model_instance in models.items():
            print(f"Training model '{model_name}' for target '{t}'...")
            m = clone(model_instance)
            
            start_train = time.time()
            m.fit(X_train_scaled, y_tr)
            end_train = time.time()
            
            start_test = time.time()
            y_pred = m.predict(X_test_scaled)
            end_test = time.time()
            
            test_set_preds[f"{t}_{model_name}_predicted"] = y_pred
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            
            results[t][model_name] = {"RMSE": rmse, "R-Squared": r2}
            trained_models[t][model_name] = m
            timing_info[t][model_name] = {
                "Training Time (s)": end_train - start_train,
                "Testing Time (s)": end_test - start_test
            }

    df_test_results = pd.DataFrame(test_set_preds, index=X_test.index)
    
    ##########          5. Predict on "inputs" DataFrame          ##########
    preds = inputs.copy()
    
    # Apply the SAME fitted transformation to the new inputs
    X_val_scaled = preproc.transform(inputs[feats])

    for t in targets:
        for model_name, m in trained_models[t].items():
            col_name = f"{model_name}_{t}_PRED"
            start_pred = time.time()
            preds[col_name] = m.predict(X_val_scaled)
            end_pred = time.time()
            timing_info[t][model_name]["Prediction Time (s)"] = end_pred - start_pred

    ##########          6. Extract Feature Importances          ##########
    # Use the preprocessor to get the correct feature names after transformation
    feat_names = preproc.get_feature_names_out()
    
    importances = {}
    for t in targets:
        importances[t] = {}
        for name, m in trained_models[t].items():
            if hasattr(m, 'coef_'):
                vals = m.coef_.ravel()
            elif hasattr(m, 'feature_importances_'):
                vals = m.feature_importances_
            else:
                continue # Skip models without importance attributes
            
            # Ensure the number of importances matches the number of feature names
            if len(vals) == len(feat_names):
                importances[t][name] = pd.Series(vals, index=feat_names)
            else:
                print(f"Warning: Mismatch between feature names ({len(feat_names)}) and importances ({len(vals)}) for model '{name}' on target '{t}'. Skipping.")


    df_importances = pd.concat(
        {t: pd.DataFrame(importances[t]) for t in importances if importances[t]},
        axis=1
    )

    return results, trained_models, preds, df_importances, timing_info, df_test_results, preproc
