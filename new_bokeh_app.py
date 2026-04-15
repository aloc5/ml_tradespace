
# import os
# import pickle
# import pandas as pd
# import numpy as np
# import glob
# import gc
# import re
# from pathlib import Path
# from bokeh.io import curdoc
# from bokeh.models import (
#     ColumnDataSource, Select, Slider, Button, DataTable, TableColumn,
#     NumberFormatter, HoverTool, TextInput, Div
# )
# from bokeh.plotting import figure
# from bokeh.layouts import column, row
# from bokeh.palettes import Category10
# from bokeh.server.server import Server
# from sklearn.neighbors import NearestNeighbors

# # =======================
# # ===== USER INPUTS =====
# # =======================
# pkl_name = "all_results_training"
# validation_metrics_name = "validation_metrics"

# DATA_DIR = Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace")
# OUTPUT_DIR = DATA_DIR / 'Final_predictions_Nick_combined' 
# METRICS_FILE = OUTPUT_DIR / f"{validation_metrics_name}.csv"

# FULL_FEATURES_LIST = [
#     'Sat Orig Total Mass (kg)', 'Sat Orig % Mass Prop (%)', 'Sat Propulsion Type', "Servicing Approach",
#     "Original Sat ISP (s)", "Original Sat Power (W)", "Original Sat Thrust (N)", 
#     "Altitude", "Design Lifetime", "Launch Vehicle", "Repositioning Prop Type", "Stakeholder"
# ]

# FEATURES = FULL_FEATURES_LIST[:7]
# CONTEXT_COLS = FULL_FEATURES_LIST[7:]

# # These variables define our axis column names
# Y_TARGET = 'MEra Avg Time-Weighted MAU'
# X_TARGET = 'MEra Avg Total Cost'
# TARGETS_TO_PLOT = [Y_TARGET, X_TARGET]

# # =======================
# # ===== LAZY LOADING SETUP =====
# # =======================
# file_map = {}
# search_pattern = str(OUTPUT_DIR / f"{pkl_name}_train*.pkl")
# split_files = glob.glob(search_pattern)

# dataset_labels_set = set()
# train_percents_set = set()

# for f_path in sorted(split_files):
#     match = re.search(r"train(\d+)", Path(f_path).name)
#     pct = float(match.group(1)) / 100.0 if match else 0.0
#     with open(f_path, "rb") as f:
#         temp_data = pickle.load(f)
#         for k in temp_data.keys():
#             file_map[k] = f_path
#             dataset_labels_set.add(k[0])
#             train_percents_set.add(k[1])
#         del temp_data

# dataset_labels = sorted(list(dataset_labels_set))
# train_percents = sorted(list(train_percents_set))

# all_data_training = {} 
# current_dataset = dataset_labels[0]
# current_train_percent = train_percents[0]

# # Initial Load
# with open(file_map[(current_dataset, current_train_percent)], "rb") as f:
#     all_data_training = pickle.load(f)

# data_entry = all_data_training[(current_dataset, current_train_percent)]
# results, trained_models, preds_df, _, _, _, preproc = data_entry["test"]
# val_df = data_entry["val_df"]

# if preds_df is not None:
#     pred_cols = [c for c in preds_df.columns if "PRED" in c or "_err" in c or "_pred_" in c.lower()]
#     val_df = val_df.join(preds_df[pred_cols], how="left", rsuffix="_preds")

# # =======================
# # ===== CONTEXT KEYS =====
# # =======================
# valid_context_cols = [c for c in CONTEXT_COLS if c in val_df.columns]
# def create_context_key(df):
#     if valid_context_cols:
#         return df.apply(lambda row: " | ".join(str(row[c]) for c in valid_context_cols), axis=1)
#     return pd.Series(["All Data"] * len(df))

# val_df["context_key"] = create_context_key(val_df)
# context_keys = sorted(val_df["context_key"].unique().tolist())
# select_context = Select(title="Select Context:", value=context_keys[0], options=context_keys)

# # =======================
# # ===== DATA SOURCES =====
# # =======================
# # Initialize sources with the target names as keys to satisfy Bokeh validation
# source_actual = ColumnDataSource(data={Y_TARGET: [], X_TARGET: []})
# new_point_source = ColumnDataSource(data={Y_TARGET: [], X_TARGET: []})
# models = list(results[TARGETS_TO_PLOT[0]].keys())
# model_sources = {m: ColumnDataSource(data={Y_TARGET: [], X_TARGET: []}) for m in models}

# # Support Metric
# X_support = preproc.transform(val_df[FEATURES + valid_context_cols])
# nn_support = NearestNeighbors(n_neighbors=10).fit(X_support)
# confidence_div = Div(text="<b>Prediction Reliability:</b> —")

# # =======================
# # ===== PLOT LOGIC ======
# # =======================
# def update_plot(attr, old, new):
#     selected_context = select_context.value
#     val_subset = val_df[val_df["context_key"] == selected_context].reset_index(drop=True)
#     n_rows = len(val_subset)
    
#     actual_data = {}
#     for t in TARGETS_TO_PLOT:
#         true_col = f"{t}_true" if f"{t}_true" in val_subset.columns else t
#         actual_data[t] = val_subset[true_col].astype(float).tolist() if true_col in val_subset.columns else [np.nan]*n_rows
#     for feat in FEATURES:
#         actual_data[feat] = val_subset[feat].fillna("None").astype(str).tolist()
#     source_actual.data = actual_data

#     for m in models:
#         new_model_data = {}
#         for t in TARGETS_TO_PLOT:
#             pred_col, err_col = f"{t}_pred_{m}", f"{t}_err_{m}"
#             new_model_data[t] = val_subset[pred_col].astype(float).tolist() if pred_col in val_subset.columns else [np.nan]*n_rows
#             new_model_data[f"{t}_err"] = val_subset[err_col].astype(float).tolist() if err_col in val_subset.columns else [np.nan]*n_rows
#         for feat in FEATURES:
#             new_model_data[feat] = val_subset[feat].fillna("None").astype(str).tolist()
#         model_sources[m].data = new_model_data

# def reload_data():
#     global val_df, results, trained_models, preproc, models, nn_support, all_data_training
#     ds, tp = select_dataset.value, float(select_train_percent.value)
#     all_data_training.clear(); gc.collect()
#     with open(file_map[(ds, tp)], "rb") as f:
#         all_data_training = pickle.load(f)
#     data_entry = all_data_training[(ds, tp)]
#     results, trained_models, preds_df, _, _, _, preproc = data_entry["test"]
#     val_df = data_entry["val_df"]
#     if preds_df is not None:
#         pred_cols = [c for c in preds_df.columns if "PRED" in c or "_err" in c or "_pred_" in c.lower()]
#         val_df = val_df.join(preds_df[pred_cols], how="left", rsuffix="_preds")
#     val_df["context_key"] = create_context_key(val_df)
#     opts = sorted(val_df["context_key"].unique().tolist())
#     select_context.options = opts
#     select_context.value = opts[0]
#     models = list(results[TARGETS_TO_PLOT[0]].keys())
#     select_model.options = models
#     select_model.value = models[0]
#     update_plot(None, None, None)

# # =======================
# # ===== UI WIDGETS ======
# # =======================
# feature_widgets = {}
# controls_list = []
# for feat in FEATURES:
#     if pd.api.types.is_numeric_dtype(val_df[feat]):
#         min_v, max_v = float(val_df[feat].min()), float(val_df[feat].max())
#         slider = Slider(title=feat, start=min_v, end=max_v, step=(max_v-min_v)/50 or 1, value=min_v)
#         txt = TextInput(value=str(min_v), title=f"{feat} (type value)")
#         slider.on_change("value", lambda a,o,n,ti=txt: ti.update(value=f"{n:.2f}"))
#         feature_widgets[feat] = slider
#         controls_list.extend([slider, txt])
#     else:
#         opts = sorted(val_df[feat].dropna().unique().tolist())
#         sel = Select(title=feat, options=opts, value=opts[0])
#         feature_widgets[feat] = sel
#         controls_list.append(sel)

# def make_prediction():
#     m_name = select_model.value
#     f_vals = {f: (w.value if isinstance(w, Select) else float(w.value)) for f, w in feature_widgets.items()}
#     ctx_parts = select_context.value.split(" | ")
#     for i, col in enumerate(valid_context_cols):
#         try:
#             f_vals[col] = pd.to_numeric(ctx_parts[i])
#         except (ValueError, TypeError):
#             f_vals[col] = ctx_parts[i] # Keep as string if numeric conversion fails

#     X_p = preproc.transform(pd.DataFrame([f_vals])[FEATURES + valid_context_cols])
#     p_vals, u_vals = {}, {}
#     for t in TARGETS_TO_PLOT:
#         m = trained_models[t][m_name]
#         p_vals[t] = float(m.predict(X_p)[0])
#         u_vals[t] = float(np.std([tree.predict(X_p) for tree in m.estimators_])) if hasattr(m, "estimators_") else 0.0
#     new_point_source.data = {**{t: [p_vals[t]] for t in TARGETS_TO_PLOT}, **{f"{t}_err": [u_vals[t]] for t in TARGETS_TO_PLOT}, **{f: [f_vals[f]] for f in FEATURES}}

# select_model = Select(title="Model", value=models[0], options=models)
# button_predict = Button(label="Predict", button_type="success")
# select_dataset = Select(title="Dataset", value=current_dataset, options=dataset_labels)
# select_train_percent = Select(title="Training %", value=str(current_train_percent), options=[str(p) for p in train_percents])

# # Attach Logic
# button_predict.on_click(make_prediction)
# select_context.on_change("value", update_plot)
# select_dataset.on_change("value", lambda a,o,n: reload_data())
# select_train_percent.on_change("value", lambda a,o,n: reload_data())

# # Setup Plot
# p = figure(title="Predicted vs Actual", x_axis_label=X_TARGET, y_axis_label=Y_TARGET, width=950, height=600)
# palette = Category10[10]
# for i, m in enumerate(models):
#     p.scatter(x=X_TARGET, y=Y_TARGET, source=model_sources[m], size=8, color=palette[i % 10], legend_label=m)
# p.scatter(x=X_TARGET, y=Y_TARGET, source=source_actual, size=16, color="black", marker="cross", legend_label="Actual")
# p.scatter(x=X_TARGET, y=Y_TARGET, source=new_point_source, size=22, color="yellow", marker="star", legend_label="Predicted")
# p.legend.click_policy = "mute"

# update_plot(None, None, None)

# manual_metrics_df = pd.read_csv(METRICS_FILE)
# metrics_table = DataTable(source=ColumnDataSource(manual_metrics_df), columns=[TableColumn(field=c, title=c) for c in manual_metrics_df.columns], width=1100, height=250)

# # Final Layout
# layout = row(column(*controls_list, select_model, button_predict, confidence_div, width=300), 
#              column(select_dataset, select_train_percent, select_context, p, metrics_table))

# def modify_doc(doc):
#     doc.add_root(layout)

# if __name__ == '__main__':
#     server = Server({'/': modify_doc}, port=5006)
#     server.start()
#     print("✅ Server running at: http://localhost:5006/")
#     server.io_loop.add_callback(server.show, "/")
#     server.io_loop.start()


import os
import pickle
import pandas as pd
import numpy as np
import glob
import gc
import re
from pathlib import Path
from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, Slider, Button, DataTable, TableColumn,
    NumberFormatter, HoverTool, TextInput, Div
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.server.server import Server
from sklearn.neighbors import NearestNeighbors

# =======================
# ===== USER INPUTS =====
# =======================
pkl_name = "ROSES_results_training"
validation_metrics_name = "ROSES_validation_metrics"

DATA_DIR = Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\ROSES_TSE_v7")#Path(r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace")
OUTPUT_DIR = DATA_DIR / 'ROSES_ML_Results_Final' 
METRICS_FILE = OUTPUT_DIR / f"{validation_metrics_name}.csv"

FULL_FEATURES_LIST = [
    "PMD_S", "PMD_Su", "Maneuverability Case", "COLA Case", "context_id"
    #'Servicing Approach', 'Sat Consumable Mass (kg)', 'Sat Aperture Size (m)', 'Sat F-Number', 'Sat Boil Off (%)', 'Sat Propulsion Type' #Bryce's Feats

    # 'Sat Orig Total Mass (kg)', 'Sat Orig % Mass Prop (%)', 'Sat Propulsion Type', "Servicing Approach",
    # #"Original Sat ISP (s)", "Original Sat Power (W)",                                                            #Nick's feats
    # "Altitude", "Design Lifetime", "Launch Vehicle", "Repositioning Prop Type", "Stakeholder"
]

FEATURES = FULL_FEATURES_LIST#[:4] # Adjusting to 6 based on your provided list indices
CONTEXT_COLS = [] #FULL_FEATURES_LIST[4:]
Y_TARGET, X_TARGET = "Benefit_MAU", "Cost_MAU" #'MEra Avg Time-Weighted MAU', 'MEra Avg Total Cost'
TARGETS_TO_PLOT = [Y_TARGET, X_TARGET]

# =======================
# ===== LAZY LOADING SETUP =====
# =======================
file_map = {}
search_pattern = str(OUTPUT_DIR / f"{pkl_name}_train*.pkl")
split_files = glob.glob(search_pattern)

dataset_labels_set, train_percents_set = set(), set()
for f_path in sorted(split_files):
    match = re.search(r"train(\d+)", Path(f_path).name)
    with open(f_path, "rb") as f:
        temp_data = pickle.load(f)
        for k in temp_data.keys():
            file_map[k] = f_path
            dataset_labels_set.add(k[0]); train_percents_set.add(k[1])
        del temp_data

dataset_labels = sorted(list(dataset_labels_set))
train_percents = sorted(list(train_percents_set))
all_data_training = {} 

# --- INITIAL STARTUP ---
current_dataset, current_train_percent = dataset_labels[0], train_percents[0]
with open(file_map[(current_dataset, current_train_percent)], "rb") as f:
    all_data_training = pickle.load(f)

data_entry = all_data_training[(current_dataset, current_train_percent)]
results, trained_models, preds_df, _, _, _, preproc = data_entry["test"]
val_df = data_entry["val_df"]

if preds_df is not None:
    pred_cols = [c for c in preds_df.columns if "PRED" in c or "_err" in c or "_pred_" in c.lower()]
    val_df = val_df.join(preds_df[pred_cols], how="left", rsuffix="_preds")

# =======================
# ===== CONTEXT SETUP =====
# =======================
valid_context_cols = [c for c in CONTEXT_COLS if c in val_df.columns]
def create_context_key(df):
    if valid_context_cols:
        return df[valid_context_cols].astype(str).apply(lambda x: " | ".join(x), axis=1)
    return pd.Series(["All Data"] * len(df), index=df.index)

val_df["context_key"] = create_context_key(val_df)
context_keys = sorted(val_df["context_key"].unique().tolist())
select_context = Select(title="Select Context:", value=context_keys[0], options=context_keys)

# Sources
source_actual = ColumnDataSource(data={Y_TARGET: [], X_TARGET: []})
new_point_source = ColumnDataSource(data={Y_TARGET: [], X_TARGET: []})
models = list(results[TARGETS_TO_PLOT[0]].keys())
model_sources = {m: ColumnDataSource(data={Y_TARGET: [], X_TARGET: []}) for m in models}

# Support Metric Initialization
X_support = preproc.transform(val_df[FEATURES + valid_context_cols])
nn_support = NearestNeighbors(n_neighbors=10).fit(X_support)
train_dists, _ = nn_support.kneighbors(X_support)

confidence_div = Div(text="<b>Prediction Reliability:</b> —")


# =======================
# ===== DEBUG PEAK ======
# =======================
print("\n🔍 --- MODEL CATEGORY CHECK ---")
for feat in FEATURES:
    if not pd.api.types.is_numeric_dtype(val_df[feat]):
        # Check raw data values
        raw_unique = val_df[feat].unique().tolist()
        print(f"Feature: {feat}")
        print(f"  > Unique values in data: {raw_unique}")
        
        # If using a standard sklearn preprocessor, try to peek at encoder categories
        try:
            # This assumes preproc is a ColumnTransformer and finds the encoder
            # If your structure is different, this might need adjustment
            for name, transformer, cols in preproc.transformers_:
                if feat in cols and hasattr(transformer, 'categories_'):
                    print(f"  > Preprocessor knows: {transformer.categories_}")
        except:
            pass
print("-------------------------------\n")

# =======================
# ===== LOGIC FUNCTIONS =
# =======================

def update_plot(attr, old, new):
    target_key = select_context.value
    print(f"🔄 Filtering plot for: {target_key}...")
    
    # EXPLICIT FILTERING
    val_subset = val_df[val_df["context_key"] == target_key].copy().reset_index(drop=True)
    n_rows = len(val_subset)
    
    if n_rows == 0:
        print("⚠️ Warning: Selected context returned 0 rows.")
        return

    # Update Actuals
    actual_data = {}
    for t in TARGETS_TO_PLOT:
        t_col = f"{t}_true" if f"{t}_true" in val_subset.columns else t
        actual_data[t] = val_subset[t_col].astype(float).tolist()
    for feat in FEATURES:
        actual_data[feat] = val_subset[feat].fillna("None").astype(str).tolist()
    source_actual.data = actual_data

    # Update Models
    for m in models:
        m_data = {}
        for t in TARGETS_TO_PLOT:
            p_col, e_col = f"{t}_pred_{m}", f"{t}_err_{m}"
            m_data[t] = val_subset[p_col].astype(float).tolist() if p_col in val_subset.columns else [np.nan]*n_rows
            m_data[f"{t}_err"] = val_subset[e_col].astype(float).tolist() if e_col in val_subset.columns else [np.nan]*n_rows
        for feat in FEATURES:
            m_data[feat] = val_subset[feat].fillna("None").astype(str).tolist()
        model_sources[m].data = m_data
    
    print(f"✅ Successfully rendered {n_rows} points.")

def reload_data():
    global val_df, results, trained_models, preproc, models, nn_support, all_data_training, train_dists
    ds, tp = select_dataset.value, float(select_train_percent.value)
    p.title.text = f"Predicted vs Actual: {tp*100}% Training Data"
    print(f"\n📂 DATA RELOAD: {ds} @ {tp*100}%")
    all_data_training.clear(); gc.collect()
    
    with open(file_map[(ds, tp)], "rb") as f:
        all_data_training = pickle.load(f)

    data_entry = all_data_training[(ds, tp)]
    results, trained_models, preds_df, _, _, _, preproc = data_entry["test"]
    val_df = data_entry["val_df"]
    
    if preds_df is not None:
        p_cols = [c for c in preds_df.columns if any(x in c for x in ["PRED", "_err", "_pred_"])]
        val_df = val_df.join(preds_df[p_cols], how="left", rsuffix="_preds")

    val_df["context_key"] = create_context_key(val_df)
    new_opts = sorted(val_df["context_key"].unique().tolist())
    select_context.options = new_opts
    select_context.value = new_opts[0]
    
    models = list(results[TARGETS_TO_PLOT[0]].keys())
    select_model.options = models
    
    X_support = preproc.transform(val_df[FEATURES + valid_context_cols])
    nn_support = NearestNeighbors(n_neighbors=10).fit(X_support)
    train_dists, _ = nn_support.kneighbors(X_support)
    
    print("🚀 Memory Swap Complete. Triggering Plot Update...")
    update_plot(None, None, None)

def make_prediction():
    m_name = select_model.value
    f_vals = {}
    
    # Handle numeric sliders (stored as tuple) and categorical selects
    for f, w in feature_widgets.items():
        if isinstance(w, tuple):  # numeric: (slider, text_input)
            slider, txt = w
            try:
                # Read from TEXT INPUT first (user's most recent input)
                f_vals[f] = float(txt.value) if txt.value else slider.value
            except ValueError:
                # Fallback to slider if text is invalid
                f_vals[f] = slider.value
        elif isinstance(w, Select):  # categorical
            f_vals[f] = w.value
        else:
            # Catch-all for any other widget type
            f_vals[f] = w.value
    
    # Sync context values to input
    ctx_parts = select_context.value.split(" | ")
    for i, col in enumerate(valid_context_cols):
        try: f_vals[col] = pd.to_numeric(ctx_parts[i])
        except: f_vals[col] = ctx_parts[i]

    X_p = preproc.transform(pd.DataFrame([f_vals])[FEATURES + valid_context_cols])
    
    p_vals, u_vals = {}, {}
    for t in TARGETS_TO_PLOT:
        m = trained_models[t][m_name]
        p_vals[t] = float(m.predict(X_p)[0])
        if hasattr(m, "estimators_"):
            u_vals[t] = float(np.std([tree.predict(X_p) for tree in m.estimators_]))
        else:
            # Use RMSE like training did
            u_vals[t] = results[t][m_name]["RMSE"]

    # Reliability Logic
    dist, _ = nn_support.kneighbors(X_p)
    density_score = 10 / (dist.sum() + 1e-8)
    train_density = 10 / (train_dists.sum(axis=1) + 1e-8)
    support_score = float(np.clip(density_score / np.median(train_density), 0, 1))

    # avg_u = np.nanmean(list(u_vals.values()))
    # u_norm = np.clip(avg_u / np.mean([val_df[t].std() for t in TARGETS_TO_PLOT]), 0, 1)
    u_scaled = [
        u_vals[t] / (val_df[t].std() + 1e-8)
        for t in TARGETS_TO_PLOT
    ]

    #u_norm = float(np.clip(np.mean(u_scaled), 0, 1)) #Hard clipping
    u_norm = 1 - np.exp(-np.mean(u_scaled)) # Soft decay transformation

    avg_u = float(np.mean(u_scaled))   # optional: redefine avg_u meaningfully

    reliability = 0.5 * support_score + 0.5 * (1 - u_norm)
    
    label = "🟢 High" if reliability > 0.8 else "🟡 Moderate" if reliability > 0.6 else "🔴 Low"
    confidence_div.text = f"<b>Reliability:</b> {label}<br>Support: {support_score:.2f} | Avg Uncertainty: {avg_u:.3f} | Norm Uncertainty: {u_norm:.2f} | Score: {reliability:.2f}"

    new_point_source.data = {**{t: [p_vals[t]] for t in TARGETS_TO_PLOT}, **{f"{t}_err": [u_vals[t]] for t in TARGETS_TO_PLOT}, **{f: [f_vals[f]] for f in FEATURES}}

# =======================
# ===== WIDGETS & UI ====
# =======================
# feature_widgets, controls_list = {}, []
# for feat in FEATURES:
#     if pd.api.types.is_numeric_dtype(val_df[feat]):
#         min_v, max_v = float(val_df[feat].min()), float(val_df[feat].max())
#         slider = Slider(title=feat, start=min_v, end=max_v, step=(max_v-min_v)/50 or 1, value=min_v)
#         txt = TextInput(value=str(min_v), title=f"{feat} (type value)")
#         slider.on_change("value", lambda a,o,n,ti=txt: ti.update(value=f"{n:.2f}"))
#         feature_widgets[feat] = slider; controls_list.extend([slider, txt])
#     else:
#         opts = sorted(val_df[feat].dropna().unique().tolist())
#         sel = Select(title=feat, options=opts, value=opts[0])
#         feature_widgets[feat] = sel; controls_list.append(sel)


# =======================
# ===== WIDGETS & UI ====
# =======================
feature_widgets, controls_list = {}, []
for feat in FEATURES:
    if pd.api.types.is_numeric_dtype(val_df[feat]):
        min_v, max_v = float(val_df[feat].min()), float(val_df[feat].max())
        slider = Slider(title=feat, start=min_v, end=max_v, step=(max_v-min_v)/50 or 1, value=min_v)
        txt = TextInput(value=str(min_v), title=f"{feat} (type value)")
        
        # Slider → Text Input (when you move slider, text updates)
        slider.on_change("value", lambda a, o, n, ti=txt: ti.update(value=f"{n:.2f}"))
        
        # Text Input → Slider (when you type in text, slider updates)
        txt.on_change("value", lambda a, o, n, sl=slider, feat_name=feat, minv=min_v, maxv=max_v: 
                      sl.update(value=np.clip(float(n) if n else minv, minv, maxv)))
        
        # Store BOTH in a tuple so make_prediction can access the text input
        feature_widgets[feat] = (slider, txt)
        controls_list.extend([slider, txt])
    else:
        opts = sorted(val_df[feat].dropna().unique().tolist())
        sel = Select(title=feat, options=opts, value=opts[0])
        feature_widgets[feat] = sel
        controls_list.append(sel)



select_model = Select(title="Model", value=models[0], options=models)
button_predict = Button(label="Predict", button_type="success")
select_dataset = Select(title="Dataset", value=current_dataset, options=dataset_labels)
select_train_percent = Select(title="Training %", value=str(current_train_percent), options=[str(p) for p in train_percents])

# Logic Attachment
button_predict.on_click(make_prediction)
select_context.on_change("value", update_plot)
select_dataset.on_change("value", lambda a,o,n: reload_data())
select_train_percent.on_change("value", lambda a,o,n: reload_data())

# Figure
p = figure(title=f"Predicted vs Actual {current_train_percent*100}%", x_axis_label=X_TARGET, y_axis_label=Y_TARGET, width=950, height=600)
palette = Category10[10]
for i, m in enumerate(models):
    p.scatter(x=X_TARGET, y=Y_TARGET, source=model_sources[m], size=8, color=palette[i % 10], legend_label=m)
p.scatter(x=X_TARGET, y=Y_TARGET, source=source_actual, size=16, color="black", marker="cross", legend_label="Actual")
# scatter for prediction point (keep reference for hover tool)
pred_renderer = p.scatter(
    x=X_TARGET,
    y=Y_TARGET,
    source=new_point_source,
    size=22,
    color="yellow",
    marker="star",
    line_color="black",
    line_width=2,
    legend_label="Predicted"
)

# add hover showing cost & MAU for the yellow/starred prediction
hover = HoverTool(
    renderers=[pred_renderer],
    tooltips=[
        (X_TARGET, f"@{{{X_TARGET}}}"),
        (Y_TARGET, f"@{{{Y_TARGET}}}")
    ]
)
p.add_tools(hover)

# Legend Customization
p.legend.click_policy = "mute"
p.add_layout(p.legend[0], 'right')
p.legend.background_fill_alpha = 0.0 # No background needed outside
p.legend.border_line_color = None

# --- AXIS LABEL SIZES ---
# This changes the "MEra Avg Total Cost" and "MEra Avg Time-Weighted MAU" text
p.xaxis.axis_label_text_font_size = "14pt"
p.yaxis.axis_label_text_font_size = "14pt"

# --- TICK MARK LABEL SIZES ---
# This changes the numbers (0.1, 0.2, etc.) along the axes
p.xaxis.major_label_text_font_size = "12pt"
p.yaxis.major_label_text_font_size = "12pt"

# Optional: Make the labels bold for your thesis presentation
p.xaxis.axis_label_text_font_style = "bold"
p.yaxis.axis_label_text_font_style = "bold"


# Initial Filtering Call
update_plot(None, None, None)

# Metrics Table
manual_metrics_df = pd.read_csv(METRICS_FILE)
metrics_table = DataTable(source=ColumnDataSource(manual_metrics_df), columns=[TableColumn(field=c, title=c) for c in manual_metrics_df.columns], width=1100, height=250)

layout = row(column(*controls_list, select_model, button_predict, confidence_div, width=300), 
             column(select_dataset, select_train_percent, select_context, p, metrics_table))

def modify_doc(doc): doc.add_root(layout)

if __name__ == '__main__':
    server = Server({'/': modify_doc}, port=5006)
    server.start()
    print("✅ Server active at: http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


