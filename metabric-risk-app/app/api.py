"""
api.py
------
Flask API serving the METABRIC-trained disease-specific mortality model.

Run:
    pip install flask joblib scikit-learn pandas numpy
    python api.py
Then POST to http://localhost:5000/predict

Educational/research-methods demonstration only. METABRIC is a research
cohort; this endpoint is not a validated clinical tool and must not be
used for real patient decision-making.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metabric_model.joblib")

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)
pipeline_full = bundle["pipeline_full"]
pipeline_clinical = bundle["pipeline_clinical"]
ALL_COLS = (
    bundle["clinical_num"] + bundle["clinical_cat"]
    + bundle["genomic_num"] + bundle["genomic_cat"]
)


def risk_category(p):
    if p < 0.20:
        return "low"
    if p < 0.40:
        return "moderate"
    if p < 0.60:
        return "high"
    return "very high"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [c for c in ALL_COLS if c not in payload]
    if missing:
        return jsonify({"error": f"missing required fields: {missing}"}), 400

    row = pd.DataFrame([payload])
    p_full = float(pipeline_full.predict_proba(row)[0, 1])
    p_clinical = float(pipeline_clinical.predict_proba(row)[0, 1])

    return jsonify({
        "disease_specific_mortality_risk_percent": round(p_full * 100, 2),
        "clinical_only_estimate_percent": round(p_clinical * 100, 2),
        "genomic_adjustment_points": round((p_full - p_clinical) * 100, 2),
        "risk_category": risk_category(p_full),
        "disclaimer": (
            "Trained on the public METABRIC research cohort for methods "
            "demonstration only. Not a validated clinical tool."
        ),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
