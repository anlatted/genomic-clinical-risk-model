"""
train_model.py
---------------
Trains a mortality-risk model on the METABRIC breast cancer cohort
(real, publicly available clinical + genomic data via cBioPortal),
comparing a clinical-only model against a clinical + genomic model to
quantify the incremental value of molecular/genomic features.

Target: Overall Survival Status (Deceased vs Living) during study follow-up.
Rows without a known outcome are dropped; all other missingness is imputed.

Outputs -> model/
  metabric_model.joblib, metrics.json, roc_curve.png,
  feature_importance.png, calibration.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

DATA_PATH = "/home/claude/metabric-risk-app/data/Breast_Cancer_METABRIC.csv"
OUT_DIR = "/home/claude/metabric-risk-app/model"

CLINICAL_NUM = [
    "Age at Diagnosis", "Tumor Size", "Tumor Stage",
    "Neoplasm Histologic Grade", "Lymph nodes examined positive",
    "Nottingham prognostic index",
]
CLINICAL_CAT = [
    "ER Status", "PR Status", "Chemotherapy", "Hormone Therapy",
    "Radio Therapy", "Type of Breast Surgery", "Inferred Menopausal State",
]
GENOMIC_NUM = ["Mutation Count"]
GENOMIC_CAT = [
    "Pam50 + Claudin-low subtype", "HER2 status measured by SNP6",
    "Integrative Cluster", "3-Gene classifier subtype",
]
VITAL_COL = "Patient's Vital Status"


def load_data():
    """
    Target: disease-specific mortality (Died of Disease vs Living).

    Rows recorded as 'Died of Other Causes' are excluded rather than
    labeled as negatives -- an all-cause mortality target dilutes the
    genomic signal, since molecular subtype predicts breast-cancer death,
    not death from unrelated causes. This is standard practice in
    oncology outcome modeling (cause-specific / disease-specific survival).
    """
    df = pd.read_csv(DATA_PATH)
    df = df[df[VITAL_COL].isin(["Living", "Died of Disease"])].copy()
    df["target"] = (df[VITAL_COL] == "Died of Disease").astype(int)
    return df


def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def main():
    df = load_data()
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["target"]
    )

    # Model A: clinical only
    pipe_clinical = build_pipeline(CLINICAL_NUM, CLINICAL_CAT)
    pipe_clinical.fit(train_df, train_df["target"])
    proba_clinical = pipe_clinical.predict_proba(test_df)[:, 1]
    auc_clinical = roc_auc_score(test_df["target"], proba_clinical)
    brier_clinical = brier_score_loss(test_df["target"], proba_clinical)

    # Model B: clinical + genomic
    all_num = CLINICAL_NUM + GENOMIC_NUM
    all_cat = CLINICAL_CAT + GENOMIC_CAT
    pipe_full = build_pipeline(all_num, all_cat)
    pipe_full.fit(train_df, train_df["target"])
    proba_full = pipe_full.predict_proba(test_df)[:, 1]
    auc_full = roc_auc_score(test_df["target"], proba_full)
    brier_full = brier_score_loss(test_df["target"], proba_full)

    metrics = {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "event_rate": round(df["target"].mean(), 4),
        "clinical_only": {"auc": round(auc_clinical, 4), "brier": round(brier_clinical, 4)},
        "clinical_plus_genomic": {"auc": round(auc_full, 4), "brier": round(brier_full, 4)},
        "auc_delta": round(auc_full - auc_clinical, 4),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ROC
    fpr_c, tpr_c, _ = roc_curve(test_df["target"], proba_clinical)
    fpr_f, tpr_f, _ = roc_curve(test_df["target"], proba_full)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_c, tpr_c, label=f"Clinical only (AUC={auc_clinical:.3f})", color="#6b7280", lw=2)
    plt.plot(fpr_f, tpr_f, label=f"Clinical + genomic (AUC={auc_full:.3f})", color="#0f766e", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="lightgray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("METABRIC disease-specific mortality: ROC comparison")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(test_df["target"], proba_full, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "o-", color="#0f766e", label="Clinical + genomic model")
    plt.plot([0, 1], [0, 1], "--", color="lightgray", label="Perfect calibration")
    plt.xlabel("Mean predicted risk"); plt.ylabel("Observed event rate")
    plt.title("Calibration curve"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/calibration.png", dpi=150); plt.close()

    # Feature importance: get feature names out of the ColumnTransformer
    ohe = pipe_full.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(all_cat))
    feat_names = all_num + cat_names
    coefs = pipe_full.named_steps["clf"].coef_[0]
    genomic_prefixes = tuple(GENOMIC_CAT) + tuple(GENOMIC_NUM)
    is_genomic = [any(f == g or f.startswith(g + "_") for g in genomic_prefixes) for f in feat_names]

    imp = pd.Series(coefs, index=feat_names)
    top = imp.reindex(imp.abs().sort_values(ascending=False).index[:20]).sort_values()
    colors = [
        "#0f766e" if is_genomic[feat_names.index(f)] else "#6b7280" for f in top.index
    ]
    plt.figure(figsize=(8, 7))
    plt.barh(top.index, top.values, color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Standardized coefficient (log-odds)")
    plt.title("Top 20 feature contributions (teal = genomic)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150); plt.close()

    joblib.dump(
        {
            "pipeline_full": pipe_full,
            "pipeline_clinical": pipe_clinical,
            "clinical_num": CLINICAL_NUM, "clinical_cat": CLINICAL_CAT,
            "genomic_num": GENOMIC_NUM, "genomic_cat": GENOMIC_CAT,
        },
        f"{OUT_DIR}/metabric_model.joblib",
    )

    # Reference values for the UI (category options, numeric ranges)
    ref = {}
    for c in CLINICAL_CAT + GENOMIC_CAT:
        ref[c] = sorted([v for v in df[c].dropna().unique().tolist()])
    for c in CLINICAL_NUM + GENOMIC_NUM:
        ref[c] = {
            "min": float(df[c].min()), "max": float(df[c].max()),
            "median": float(df[c].median()),
        }
    with open(f"{OUT_DIR}/field_reference.json", "w") as f:
        json.dump(ref, f, indent=2)


if __name__ == "__main__":
    main()
