
# shap_plotly_app_updated.py
import pickle
import pandas as pd
import numpy as np
import shap
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import tempfile
import os
import plotly.io as pio

# ================= CONFIG =================
SHAP_RESULTS_FILE = r"C:\Users\locri\OneDrive - Massachusetts Institute of Technology\Desktop\Thesis\EO Tradespace\Final_predictions_Nick_combined\shap_results.pkl"  # path to the pickle file with SHAP results

# ================= LOAD SHAP RESULTS =================
with open(SHAP_RESULTS_FILE, "rb") as f:
    shap_results = pickle.load(f)

if not shap_results:
    raise RuntimeError("❌ No SHAP results found. Run shap_analysis.py first.")

dataset_nums = sorted(set(k[0] for k in shap_results.keys()))
train_percents = sorted(set(k[1] for k in shap_results.keys()))
print(f"✅ Loaded SHAP results for {len(shap_results)} dataset/train combinations")
print(f"   Datasets: {dataset_nums}")
print(f"   Training fractions: {train_percents}")

# Global storage for the current plot (for download)
current_plot = {"fig": None, "type": None, "filename": None}

# ================= DASH APP =================
app = Dash(__name__)
app.title = "SHAP Feature Importance Explorer"

# ================= DASH LAYOUT =================
app.layout = html.Div([
    html.H1("SHAP Feature Importance Explorer"),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-select',
            options=[{'label': str(d), 'value': d} for d in dataset_nums],
            value=dataset_nums[0]
        )
    ], style={'width':'300px'}),

    html.Div([
        html.Label("Select Training %:"),
        dcc.Dropdown(
            id='train-percent-select',
            options=[{'label': f"{int(p*100)}%", 'value': p} for p in train_percents],
            value=train_percents[0]
        )
    ], style={'width':'300px', 'margin-top':'10px'}),

    html.Div([
        html.Label("Select Target:"),
        dcc.Dropdown(id='target-select')
    ], style={'width':'300px', 'margin-top':'10px'}),

    html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(id='model-select')
    ], style={'width':'300px', 'margin-top':'10px'}),

    dcc.Tabs(id='tabs', value='tab-mean', children=[
        dcc.Tab(label='Mean |SHAP|', value='tab-mean'),
        dcc.Tab(label='Beeswarm Summary', value='tab-beeswarm'),
        dcc.Tab(label='Dependence Plot', value='tab-dependence'),
        dcc.Tab(label='Force Plot', value='tab-force')
    ]),

    html.Div(id='tab-content', style={'width': '95%', 'margin': '20px auto'}),

    html.Div([
        html.Label("Dependence Plot Feature:"),
        dcc.Dropdown(id='dep-feature-select')
    ], style={'width':'300px', 'margin-top':'20px'}),

    html.Div([
        html.Label("Force Plot Sample Index:"),
        dcc.Input(id='force-sample', type='number', value=0, min=0, step=1)
    ], style={'width':'300px', 'margin-top':'10px'}),

    html.Button("Download Current Plot", id='download-button', n_clicks=0, style={'margin-top': '20px'}),
    dcc.Download(id="download")
])

# ================= CALLBACKS =================
@app.callback(
    Output('target-select', 'options'),
    Output('target-select', 'value'),
    Input('dataset-select', 'value'),
    Input('train-percent-select', 'value')
)
def update_targets(dataset_num, train_percent):
    if dataset_num is None or train_percent is None:
        return [], None
    dataset_key = (dataset_num, train_percent)
    if dataset_key not in shap_results:
        return [], None
    targets = list(shap_results[dataset_key].keys())
    options = [{'label': t, 'value': t} for t in targets]
    value = targets[0] if targets else None
    return options, value

@app.callback(
    Output('model-select', 'options'),
    Output('model-select', 'value'),
    Input('dataset-select', 'value'),
    Input('train-percent-select', 'value'),
    Input('target-select', 'value')
)
def update_models(dataset_num, train_percent, target_name):
    if not target_name or dataset_num is None or train_percent is None:
        return [], None
    dataset_key = (dataset_num, train_percent)
    if dataset_key not in shap_results or target_name not in shap_results[dataset_key]:
        return [], None
    models = list(shap_results[dataset_key][target_name].keys())
    options = [{'label': m, 'value': m} for m in models]
    value = models[0] if models else None
    return options, value

@app.callback(
    Output('dep-feature-select', 'options'),
    Output('dep-feature-select', 'value'),
    Input('dataset-select', 'value'),
    Input('train-percent-select', 'value'),
    Input('target-select', 'value'),
    Input('model-select', 'value')
)
def update_dep_features(dataset_num, train_percent, target_name, model_name):
    if not model_name or not target_name or dataset_num is None or train_percent is None:
        return [], None
    dataset_key = (dataset_num, train_percent)
    if dataset_key not in shap_results or target_name not in shap_results[dataset_key]:
        return [], None
    shap_dict = shap_results[dataset_key][target_name][model_name]
    features_df = shap_dict.get("features_raw", None)
    if features_df is None:
        # fallback: use X_test as DataFrame
        features_df = pd.DataFrame(shap_dict["X_test"], columns=shap_dict["feature_names"])
    options = [{'label': c, 'value': c} for c in features_df.columns]
    value = features_df.columns[0] if len(features_df.columns) > 0 else None
    return options, value

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('dataset-select', 'value'),
    Input('train-percent-select', 'value'),
    Input('target-select', 'value'),
    Input('model-select', 'value'),
    Input('dep-feature-select', 'value'),
    Input('force-sample', 'value')
)
def render_tab(tab, dataset_num, train_percent, target_name, model_name, dep_feature, force_sample):
    if not target_name or not model_name or dataset_num is None or train_percent is None:
        return html.Div("No SHAP data for this selection")

    # Get SHAP dict
    dataset_key = (dataset_num, train_percent)
    if dataset_key not in shap_results or target_name not in shap_results[dataset_key]:
        return html.Div("No SHAP data for this selection")
    
    shap_dict = shap_results[dataset_key][target_name][model_name]
    shap_values = shap_dict["shap_values"]  # numpy array
    encoded_feature_names = shap_dict["feature_names"]
    X_encoded = shap_dict["X_test"]

    # ========== Mean |SHAP| ==========

    if tab == 'tab-mean':
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        df_plot = pd.DataFrame({
            "Feature": encoded_feature_names,
            "MeanAbsSHAP": mean_abs_shap
        })

        fig = px.bar(
            df_plot,
            y="Feature",
            x="MeanAbsSHAP",
            orientation='h',
            title=f"Mean |SHAP| - {model_name}"
        )

        # Automatically scale figure height based on number of features
        n_features = len(encoded_feature_names)

        fig.update_layout(
            yaxis={'autorange':'reversed'},
            height=max(400, n_features * 35),   # dynamic height
            yaxis_tickfont=dict(size=14),       # feature label font
            xaxis_tickfont=dict(size=14),       # x-axis font
            title_font=dict(size=20)
        )

        # Optional: show numeric SHAP value labels
        fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')

        current_plot.update({"fig": fig, "type": "plotly", "filename": f"mean_shap_{model_name}.html"})
        return dcc.Graph(figure=fig)

    # ========== Beeswarm ==========
    elif tab == 'tab-beeswarm':
        df_shap = pd.DataFrame(shap_values, columns=encoded_feature_names)
        df_features_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
        x_vals, y_vals, color_vals, hover_text = [], [], [], []

        for col in encoded_feature_names:
            x_vals.extend(df_shap[col].tolist())
            y_vals.extend([col] * len(df_shap))
            color_vals.extend(df_features_encoded[col].tolist())
            hover_text.extend(df_features_encoded[col].round(3).astype(str).tolist())

        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                color=color_vals,
                colorbar=dict(title="Feature Value"),
                colorscale='Viridis',
                showscale=True,
                size=6,
                opacity=0.7
            )
        ))
        fig.update_layout(
            showlegend=False,
            yaxis={'autorange':'reversed'},
            xaxis_title="SHAP value (impact on model output)",
            title=f"SHAP Beeswarm Summary - {model_name}",
            margin=dict(l=150, r=50, t=50, b=50)
        )

        current_plot.update({"fig": fig, "type": "plotly", "filename": f"beeswarm_{model_name}.html"})
        return dcc.Graph(figure=fig)

    # ========== Dependence Plot ==========
    elif tab == 'tab-dependence':
        if dep_feature is None:
            return html.Div("Select a feature for dependence plot")
        
        # Try to use raw features if available
        features_raw = shap_dict.get("features_raw", None)
        if features_raw is not None and dep_feature in features_raw.columns:
            # Use raw feature directly from features_raw
            feature_vals = features_raw[dep_feature].values
            # Find encoded columns that match this feature
            matching_cols = [i for i, name in enumerate(encoded_feature_names) if dep_feature in name]
            if matching_cols:
                shap_vals = shap_values[:, matching_cols].sum(axis=1)
            else:
                shap_vals = shap_values.sum(axis=1)  # fallback: sum all
        else:
            # Fallback: try to match in encoded features
            matching_cols = [i for i, name in enumerate(encoded_feature_names) if dep_feature in name]
            if not matching_cols:
                return html.Div(f"Feature '{dep_feature}' not found in SHAP data")
            shap_vals = shap_values[:, matching_cols].sum(axis=1)
            feature_vals = X_encoded[:, matching_cols].sum(axis=1)
        
        fig = px.scatter(x=feature_vals, y=shap_vals,
                         title=f"Dependence Plot - {dep_feature}",
                         labels={"x": dep_feature, "y": "SHAP Value"})
        current_plot.update({"fig": fig, "type": "plotly", "filename": f"dependence_{dep_feature}_{model_name}.html"})
        return dcc.Graph(figure=fig)

    # ========== Force/Waterfall ==========
    elif tab == 'tab-force':
        if force_sample is None:
            force_sample = 0
        sample_idx = min(int(force_sample), shap_values.shape[0]-1)
        # create shap.Explanation for single row
        expl = shap.Explanation(values=shap_values[sample_idx],
                                base_values=np.mean(shap_values[sample_idx]),  # fallback base
                                data=X_encoded[sample_idx],
                                feature_names=encoded_feature_names)

        # Matplotlib waterfall
        fig, ax = plt.subplots(figsize=(8, max(4, len(encoded_feature_names)*0.4)))
        shap.plots.waterfall(expl, show=False)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_bytes = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        current_plot.update({"fig": fig, "type": "matplotlib", "filename": f"force_plot_{model_name}.png"})
        return html.Img(src=f"data:image/png;base64,{img_bytes}",
                        style={'width':'100%', 'height':'auto', 'border':'1px solid #ccc'})

    return html.Div("Unknown tab")

# ================= DOWNLOAD CALLBACK =================
@app.callback(
    Output("download", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_plot(n_clicks):
    if not current_plot["fig"]:
        return
    fig = current_plot["fig"]
    plot_type = current_plot["type"]
    filename = current_plot["filename"]
    tmp_path = os.path.join(tempfile.gettempdir(), filename)

    if plot_type == "plotly":
        pio.write_html(fig, file=tmp_path, auto_open=False)
        return dcc.send_file(tmp_path)
    elif plot_type == "matplotlib":
        fig.savefig(tmp_path, format='png', bbox_inches='tight')
        return dcc.send_file(tmp_path)

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
