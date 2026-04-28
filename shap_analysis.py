# import pandas as pd
# import shap
# import pickle
# from pathlib import Path
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# import os

# # ================= CONFIG =================
# PKL_PATH = "all_data_training.pkl"
# RAW_DATA_DIR = Path("raw_data")  # "raw_data" or "raw_data_new"
# RAW_DATA_PREFIX = "raw_data_" if RAW_DATA_DIR.name == "raw_data" else "Multi_Era_df_lhs_"
# TARGET_NAME = "MEra Fuzzy Pareto Number (FPN)" #"MEra Avg Time-Weighted MAU"  # Target for SHAP
# SHAP_OUTFILE = "shap_results.pkl"

# # Categorical and numeric feature names (must match your ML training)
# categorical_features = ['Sat Propulsion Type', 'Servicing Approach']
# numeric_features = [
#     'Sat Orig Total Mass (kg)', 'Sat Orig % Mass Prop (%)',
#     'Original Sat Power (W)', 'Original Sat Thrust (N)'
# ]

# # ================= LOAD TRAINED RESULTS =================
# if not os.path.exists(PKL_PATH) or os.path.getsize(PKL_PATH) == 0:
#     print(f"❌ ERROR: Pickle file not found or is empty at {PKL_PATH}")
#     exit()

# try:
#     with open(PKL_PATH, "rb") as f:
#         all_results = pickle.load(f)
# except EOFError:
#     print(f"❌ ERROR: Pickle file at {PKL_PATH} seems empty or corrupted (EOF).")
#     exit()

# print(f"✅ Loaded {len(all_results)} model results from {PKL_PATH}")

# # ================= SHAP RESULTS STORAGE =================
# shap_results = {}


# # ================= LOOP THROUGH DATASETS =================
# for (dataset_num, train_percent), result_dict in all_results.items():
#     try:
#         # Check structure
#         if not isinstance(result_dict, dict) or "test" not in result_dict:
#             print(f"⚠️ Unexpected structure for dataset {dataset_num}, skipping.")
#             continue

#         # Extract inner tuple and feature list
#         result_test = result_dict["test"]
#         features_used = result_dict.get("features", [])
#         val_df = result_dict.get("val_df", None)

#         # Unpack tuple from "test"
#         if len(result_test) < 6:
#             print(f"⚠️ Unexpected 'test' tuple length for dataset {dataset_num}. Skipping.")
#             continue

#         results, trained_models, preds_with_design, df_importances, timing_info, test_results_df = result_test[:6]

#         # Load raw dataset
#         file_path = RAW_DATA_DIR / f"{RAW_DATA_PREFIX}{dataset_num}.csv"
#         if not file_path.exists():
#             print(f"❌ File not found: {file_path}")
#             continue

#         raw_df = pd.read_csv(file_path)

#         # Recreate train/test split to get test_df
#         train_df, test_df = train_test_split(raw_df, test_size=(1 - train_percent), random_state=123)

#         # Validate feature existence
#         if not all(f in test_df.columns for f in features_used):
#             print(f"❌ Missing some feature columns in dataset {dataset_num}, skipping...")
#             continue

#         X_test_raw = test_df[features_used]

#         # ================= ENCODE FEATURES =================
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#                 ('num', 'passthrough', numeric_features)
#             ]
#         )
#         preprocessor.fit(raw_df[categorical_features + numeric_features])
#         X_test = preprocessor.transform(X_test_raw)

#         # ================= LOOP THROUGH MODELS =================
#         shap_results[(dataset_num, train_percent)] = {}
#         if TARGET_NAME not in trained_models:
#             print(f"⚠️ Target '{TARGET_NAME}' not found in trained_models for dataset {dataset_num}.")
#             continue

#         for model_name, model in trained_models[TARGET_NAME].items():
#             print(f"🔍 SHAP for Dataset {dataset_num} - {int(train_percent*100)}% - {model_name}")
#             try:
#                 explainer = shap.Explainer(model.predict, X_test)
#                 shap_values = explainer(X_test)

#                 shap_results[(dataset_num, train_percent)][model_name] = {
#                     "shap_values": shap_values,
#                     "features_raw": X_test_raw,
#                     "explainer": explainer,
#                     "model": model
#                 }

#             except Exception as e:
#                 print(f"❌ SHAP failed for {model_name} on dataset {dataset_num}: {e}")
#                 continue

#     except Exception as e:
#         print(f"❌ Error processing dataset {dataset_num}, train {train_percent}: {e}")
#         continue



# # ================= SAVE SHAP RESULTS =================
# with open(SHAP_OUTFILE, "wb") as f:
#     pickle.dump(shap_results, f)

# print(f"✅ SHAP results saved to: {SHAP_OUTFILE} ({len(shap_results)} dataset entries)")


########### SHAP CODE FOR THE COMBINED DATA ###########


# import pandas as pd
# import shap
# import pickle
# from pathlib import Path
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# import os
# import numpy as np

# # ================= CONFIG =================
# PKL_PATH = "all_data_training.pkl"
# RAW_DATA_DIR = Path("raw_data")  # folder containing raw CSVs
# RAW_DATA_PREFIX = "raw_data_" if RAW_DATA_DIR.name == "raw_data" else "Multi_Era_df_lhs_"
# TARGET_NAME = "MEra Fuzzy Pareto Number (FPN)"  # target for SHAP
# SHAP_OUTFILE = "shap_results.pkl"

# # PKL_PATH = "C:\\Users\\locri\\OneDrive - Massachusetts Institute of Technology\\Desktop\\Thesis\\EO Tradespace\\Roses_TSE_v7\\validation_results.pkl"
# # RAW_DATA_DIR = "C:\\Users\\locri\\OneDrive - Massachusetts Institute of Technology\\Desktop\\Thesis\\EO Tradespace\\Roses_TSE_v7"  # folder containing raw CSVs
# # RAW_DATA_PREFIX = "raw_data_" if RAW_DATA_DIR.name == "raw_data" else "Multi_Era_df_lhs_"
# # TARGET_NAME = "MEra Fuzzy Pareto Number (FPN)"  # target for SHAP
# # SHAP_OUTFILE = "ROSES_shap_results.pkl"


# categorical_features = ['Sat Propulsion Type', 'Servicing Approach', 'Launch Vehicle', 'Repositioning Prop Type', 'Stakeholder', 'Altitude']
# numeric_features = [
#     'Sat Orig Total Mass (kg)', 'Sat Orig % Mass Prop (%)',
#     'Original Sat Power (W)', 'Original Sat Thrust (N)', 'Design Lifetime'
# ]

# # ================= LOAD TRAINED RESULTS =================
# if not os.path.exists(PKL_PATH) or os.path.getsize(PKL_PATH) == 0:
#     raise FileNotFoundError(f"Pickle file not found or empty: {PKL_PATH}")

# with open(PKL_PATH, "rb") as f:
#     all_results = pickle.load(f)

# print(f"✅ Loaded {len(all_results)} model results from {PKL_PATH}")

# # ================= SHAP RESULTS STORAGE =================
# shap_results = {}

# # ================= LOOP THROUGH DATASETS =================
# for (dataset_num, train_percent), result_dict in all_results.items():
#     try:
#         # Extract the "test" tuple and other info
#         if "test" not in result_dict:
#             print(f"⚠️ Dataset {dataset_num} missing 'test' key, skipping.")
#             continue

#         result_test = result_dict["test"]
#         val_df = result_dict.get("val_df", None)
#         features_used = result_dict.get("features", [])

#         # Unpack test tuple
#         if len(result_test) < 6:
#             print(f"⚠️ Dataset {dataset_num} has unexpected 'test' tuple length, skipping.")
#             continue

#         results, trained_models, preds_df, df_importances, timing_info, test_results_df, preproc = result_test[:7]

#         # Load raw CSV
#         file_path = RAW_DATA_DIR / f"{RAW_DATA_PREFIX}{dataset_num}.csv"
#         if not file_path.exists():
#             print(f"❌ File not found: {file_path}")
#             continue

#         raw_df = pd.read_csv(file_path)

#         # Ensure train_percent is fraction
#         if train_percent > 1:
#             train_percent = train_percent / 100

#         # Train/test split to match original dataset
#         train_df, test_df = train_test_split(raw_df, test_size=(1 - train_percent), random_state=123)

#         # Validate features
#         missing_features = [f for f in features_used if f not in test_df.columns]
#         if missing_features:
#             print(f"❌ Dataset {dataset_num} missing features {missing_features}, skipping...")
#             continue

#         X_test_raw = test_df[features_used]

#         # ================= USE PREPROCESSOR FROM TRAINING =================
#         X_test = preproc.transform(X_test_raw)
#         encoded_feature_names = preproc.get_feature_names_out()

#         # Debugging: Log raw and preprocessed features
#         print("🔍 Debugging SHAP Computation Pipeline")
#         print("Raw feature names:")
#         print(X_test_raw.columns.tolist())
#         print("Preprocessed feature names:")
#         print(encoded_feature_names)
#         print("Shape of raw features:", X_test_raw.shape)
#         print("Shape of preprocessed features:", X_test.shape)

#         # ================= LOOP THROUGH MODELS =================
#         shap_results[(dataset_num, train_percent)] = {}
#         if TARGET_NAME not in trained_models:
#             print(f"⚠️ Target '{TARGET_NAME}' not found in trained_models for dataset {dataset_num}.")
#             continue

#         for model_name, model in trained_models[TARGET_NAME].items():
#             print(f"🔍 SHAP for Dataset {dataset_num} - {int(train_percent*100)}% - {model_name}")
#             try:
#                 explainer = shap.Explainer(model.predict, X_test)
#                 shap_values = explainer(X_test)

#                 shap_results[(dataset_num, train_percent)][model_name] = {
#                     "shap_values": shap_values,
#                     "features_raw": X_test_raw,
#                     "encoded_feature_names": encoded_feature_names,
#                     "explainer": explainer,
#                     "model": model
#                 }

#             except Exception as e:
#                 print(f"❌ SHAP failed for {model_name} on dataset {dataset_num}: {e}")
#                 continue

#     except Exception as e:
#         print(f"❌ Error processing dataset {dataset_num}: {e}")
#         continue

# # ================= SAVE SHAP RESULTS =================
# with open(SHAP_OUTFILE, "wb") as f:
#     pickle.dump(shap_results, f)

# print(f"✅ SHAP results saved to: {SHAP_OUTFILE} ({len(shap_results)} dataset entries)")



########### SHAP ANALYSIS FOR ml_tradespace STRUCTURE (subset) ###########

import pandas as pd
import shap
import pickle
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time


# ================= CONFIG =================
#PKL_PATH = Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\Bryce_data\ML_Results_2")
RAW_DATA_DIR = Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\ROSES_TSE_v7")  # folder containing raw CSV
RAW_DATA_PREFIX = "context_evaluations_with_MAUs"  # matches your new training data
SHAP_OUTFILE = Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\ROSES_TSE_v7\ROSES_ML_Results_Final\ROSES_shap_results.pkl")  # where to save SHAP results
MAX_ROWS = 1000  # Use only this many rows per test set

# ================= LOAD TRAINED RESULTS =================
# if not os.path.exists(PKL_PATH) or os.path.getsize(PKL_PATH) == 0:
#     raise FileNotFoundError(f"Pickle file not found or empty: {PKL_PATH}")

# with open(PKL_PATH, "rb") as f:
#     all_results = pickle.load(f)

# print(f"✅ Loaded {len(all_results)} dataset entries from {PKL_PATH}")
# print(f"Sample keys: {list(all_results.keys())[:3]}")
from pathlib import Path
import pickle

# --------------------------------------------------------
# read every “trainXX” pickle and append into all_results
# --------------------------------------------------------
OUTPUT_DIR = Path(
    r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\ROSES_TSE_v7\ROSES_ML_Results_Final"#Bryce_data\ML_Results_2"
)
pattern = "ROSES_results_training_train*.pkl" #"Bryce_results_training_train*.pkl"

all_results = {}
for pkl_path in sorted(OUTPUT_DIR.glob(pattern)):
    print("loading", pkl_path.name)
    with open(pkl_path, "rb") as f:
        part = pickle.load(f)
    all_results.update(part)           # just extend the master dict
    # (optionally) del part to free the chunk after copying
    del part

print(f"✅ all_results now contains {len(all_results)} entries")
# --------------------------------------------------------
# from here on your existing SHAP loop can remain unchanged
# --------------------------------------------------------



# ================= STORAGE =================
shap_results = {}

# ================= LOOP THROUGH DATASETS =================
for (dataset_num, train_frac), result_dict in all_results.items():

    print(f"\n🚀 Processing Dataset {dataset_num} | Train Fraction {train_frac}")

    try:
        if "test" not in result_dict:
            print("⚠️ Missing 'test' key. Skipping.")
            continue

        # Unpack ml_tradespace return tuple
        (
            results,
            trained_models,
            preds,
            df_importances,
            timing_info,
            df_test_results,
            preproc
        ) = result_dict["test"]

        feats = result_dict["features"]

        # Load matching raw CSV
        file_path = RAW_DATA_DIR / f"{RAW_DATA_PREFIX}.csv"  # master_tradespace.csv
        if not file_path.exists():
            print(f"❌ Raw file not found: {file_path}. Trying alternate names...")
            # Fallback: try to find any CSV with the dataset name
            matches = list(RAW_DATA_DIR.glob(f"*{dataset_num}*.csv"))
            if matches:
                file_path = matches[0]
                print(f"   Found alternate file: {file_path}")
            else:
                print(f"❌ No file found for dataset {dataset_num}. Skipping.")
                continue

        raw_df = pd.read_csv(file_path)

        # Recreate train/test split exactly
        if train_frac > 1:
            train_frac = train_frac / 100

        train_df, test_df = train_test_split(
            raw_df,
            test_size=(1 - train_frac),
            random_state=123
        )

        X_test_raw = test_df[feats].fillna("None")

        # Subset to MAX_ROWS for faster SHAP
        if len(X_test_raw) > MAX_ROWS:
            X_test_raw = X_test_raw.sample(n=MAX_ROWS, random_state=123)

        # Apply stored preprocessing
        X_test = preproc.transform(X_test_raw)
        feature_names = preproc.get_feature_names_out()

        shap_results[(dataset_num, train_frac)] = {}

        # ================= LOOP THROUGH TARGETS =================
        for target_name, model_dict in trained_models.items():

            shap_results[(dataset_num, train_frac)][target_name] = {}

            # ================= LOOP THROUGH MODELS =================
            for model_name, model in model_dict.items():

                print(f"   🔍 SHAP → {target_name} | {model_name}")

                try:
                    # Tree-based models
                    if hasattr(model, "estimators_") or hasattr(model, "feature_importances_"):
                        print("      Using TreeExplainer for tree-based model")
                        start_time = time.time()
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_test, approximate=True)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        end_time = time.time()
                        print(f"      SHAP computed in {end_time - start_time:.2f} seconds")

                    else:
                        # Non-tree models
                        print("      Using KernelExplainer for non-tree model")
                        start_time = time.time()
                        explainer = shap.Explainer(model.predict, X_test)
                        shap_vals = explainer(X_test).values
                        end_time = time.time()
                        print(f"      SHAP computed in {end_time - start_time:.2f} seconds")

                    print(f"      SHAP computed successfully for {model_name} on target {target_name}")

                    # Store everything needed for the Dash app
                    shap_results[(dataset_num, train_frac)][target_name][model_name] = {
                        "shap_values": shap_vals,
                        "feature_names": feature_names,
                        "X_test": X_test,
                        "features_raw": X_test_raw,              # unencoded features
                        "encoded_feature_names": feature_names   # names after preprocessing
                    }

                except Exception as e:
                    print(f"   ❌ SHAP failed: {e}")
                    continue

    except Exception as e:
        print(f"❌ Dataset {dataset_num} failed: {e}")
        continue

# ================= SAVE RESULTS =================
with open(SHAP_OUTFILE, "wb") as f:
    pickle.dump(shap_results, f)

print(f"\n✅ SHAP results saved to {SHAP_OUTFILE}")
print(f"Total dataset entries processed: {len(shap_results)}")
