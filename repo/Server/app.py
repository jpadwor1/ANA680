from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app) 
DATA_PATH = os.path.join(os.path.dirname(__file__), "breast-cancer-wisconsin-data1.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_rbf_top_features.pkl")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

def load_and_clean_df(path):
    df = pd.read_csv(path).copy()
    if "bare_nuclei" in df.columns:
        df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"].replace("?", np.nan))
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df

FEATURE_HELP = {
    "clump_thickness": "How thick the cell clumps appear (1–10).",
    "uniformity_of_cell_size": "How uniform cell sizes look (1–10).",
    "uniformity_of_cell_shape": "How uniform cell shapes look (1–10).",
    "marginal_adhesion": "How strongly cells stick at the edges (1–10).",
    "single_epithelial_cell_size": "Size of single epithelial cells (1–10).",
    "bare_nuclei": "How many nuclei appear without surrounding cell material (1–10).",
    "bland_chromatin": "Texture/appearance of chromatin in the nucleus (1–10).",
    "normal_nucleoli": "Prominence of nucleoli inside the nucleus (1–10).",
    "mitoses": "How often cells appear to be dividing (1–10).",
}

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

SEL = bundle["selected_features"]
IMPUTER = bundle["imputer"]
SCALER = bundle["scaler"]
MODEL = bundle["model"]

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/model-info")
def model_info():
    df = load_and_clean_df(DATA_PATH)

    y = df["class"].map({2: 0, 4: 1})
    X = df.drop(columns=["class"])

    X_sel = X[SEL].copy()

    for c in X_sel.columns:
        X_sel[c] = pd.to_numeric(X_sel[c], errors="coerce")

    X_sel = X_sel.fillna(X_sel.median(numeric_only=True))

    ranges = {}
    for f in SEL:
        ranges[f] = {
            "min": float(X_sel[f].min()),
            "max": float(X_sel[f].max()),
        }

    benign_median = X_sel[y == 0].median(numeric_only=True).to_dict()
    malignant_median = X_sel[y == 1].median(numeric_only=True).to_dict()

    rng = np.random.default_rng(42)
    benign_idx = X_sel[y == 0].sample(1, random_state=42).index[0]
    malig_idx = X_sel[y == 1].sample(1, random_state=42).index[0]

    presets = {
        "benign_typical": {f: float(benign_median[f]) for f in SEL},
        "malignant_typical": {f: float(malignant_median[f]) for f in SEL},
        "benign_sample": {f: float(X_sel.loc[benign_idx, f]) for f in SEL},
        "malignant_sample": {f: float(X_sel.loc[malig_idx, f]) for f in SEL},
    }

    labels = {f: f.replace("_", " ").title() for f in SEL}
    helptext = {f: FEATURE_HELP.get(f, "Value from 1–10.") for f in SEL}

    return jsonify({
        "selected_features": SEL,
        "ranges": ranges,
        "presets": presets,
        "labels": labels,
        "helptext": helptext,
        "note": "Examples are for testing the model only and are not medical advice."
    })

@app.post("/predict")
def predict():
    """
    Expected JSON:
    {
      "features": {
        "clump_thickness": 5,
        "bare_nuclei": 1,
        ...
      }
    }
    """
    data = request.get_json(silent=True) or {}
    feats = data.get("features", {})

    row = {}
    missing = []
    for f in SEL:
        val = feats.get(f, None)
        if val is None or val == "":
            missing.append(f)
        row[f] = val

    if missing:
        return jsonify({"error": f"Missing feature values: {missing}"}), 400

    X = pd.DataFrame([row], columns=SEL)
    for col in SEL:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().iloc[0]].tolist()
        return jsonify({"error": f"Non-numeric or invalid values for: {bad_cols}"}), 400

    X_imp = IMPUTER.transform(X)
    X_scaled = SCALER.transform(X_imp)
    pred = int(MODEL.predict(X_scaled)[0])

    return jsonify({
        "prediction": pred,
        "label": "Malignant" if pred == 1 else "Benign",
        "used_features": SEL
    })

@app.get("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.get("/<path:path>")
def serve_static_files(path):
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.exists(file_path):
        return send_from_directory(STATIC_DIR, path)

    return send_from_directory(STATIC_DIR, "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
