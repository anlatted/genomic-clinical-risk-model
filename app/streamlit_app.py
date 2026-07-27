"""
streamlit_app.py
-----------------
Interactive front end for the METABRIC disease-specific mortality model.

Unlike a JS/React demo, this runs the ACTUAL trained sklearn pipeline
directly -- no coefficients need to be re-implemented or kept in sync.

Run:
    pip install streamlit joblib scikit-learn pandas numpy matplotlib
    streamlit run streamlit_app.py
"""
import os
import json

import joblib
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "model", "metabric_model.joblib")
REF_PATH = os.path.join(_HERE, "..", "model", "field_reference.json")
METRICS_PATH = os.path.join(_HERE, "..", "model", "metrics.json")

st.set_page_config(
    page_title="Breast Cancer Outcome Estimator",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    with open(REF_PATH) as f:
        ref = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return ref, metrics


bundle = load_model()
pipeline_full = bundle["pipeline_full"]
pipeline_clinical = bundle["pipeline_clinical"]
ref, metrics = load_reference()

GENOMIC_CAT = bundle["genomic_cat"]
GENOMIC_NUM = bundle["genomic_num"]
CLINICAL_CAT = bundle["clinical_cat"]
CLINICAL_NUM = bundle["clinical_num"]

PAM50_INFO = {
    "LumA": "Luminal A — ER+, typically slower-growing, best prognosis",
    "LumB": "Luminal B — ER+, more proliferative than Luminal A",
    "Her2": "HER2-enriched — HER2-driven, historically aggressive",
    "Basal": "Basal-like — largely overlaps with triple-negative",
    "claudin-low": "Claudin-low — low cell-adhesion markers, mesenchymal features",
    "Normal": "Normal-like — expression pattern close to normal breast tissue",
    "NC": "Not classified",
}


def risk_category(p):
    if p < 0.20:
        return "Low", "#1f9d6f"
    if p < 0.40:
        return "Moderate", "#d9a412"
    if p < 0.60:
        return "High", "#e0692a"
    return "Very high", "#c23b3b"


# ---------------------------------------------------------------------
# Custom styling — mirrors the React demo's palette (teal genomics accent,
# IBM Plex fonts) so both front ends feel like the same product.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
    .mono { font-family: 'IBM Plex Mono', monospace; }
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        letter-spacing: 0.16em; text-transform: uppercase;
        color: #0f766e; font-weight: 600;
    }
    .genomic-panel {
        background-color: #12201c; border-radius: 10px; padding: 18px 20px;
    }
    .genomic-panel label, .genomic-panel .stMarkdown { color: #c8d8d3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">METABRIC cohort · n=1,483 · research demo</div>', unsafe_allow_html=True)
st.title("Breast Cancer Outcome Estimator")
st.caption(
    "Disease-specific mortality model trained on real clinical and molecular "
    "data from the METABRIC study, comparing clinical factors alone against "
    "clinical + genomic subtyping."
)

with st.expander("Model performance (held-out test set)"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Clinical-only AUC", f"{metrics['clinical_only']['auc']:.3f}")
    c2.metric("Clinical + genomic AUC", f"{metrics['clinical_plus_genomic']['auc']:.3f}",
              delta=f"+{metrics['auc_delta']:.3f}")
    c3.metric("Test set size", f"{metrics['n_test']} patients")

st.divider()

col_form, col_result = st.columns([1.15, 0.85], gap="large")

with col_form:
    st.subheader("Clinical & pathological")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age at diagnosis", 20, 100, 61)
    tumor_size = c2.number_input("Tumor size (mm)", 1, 180, 28)
    tumor_stage = c3.number_input("Tumor stage", 0, 4, 2)

    c1, c2, c3 = st.columns(3)
    grade = c1.number_input("Histologic grade", 1, 3, 3)
    nodes_pos = c2.number_input("Positive lymph nodes", 0, 45, 3)
    npi = c3.number_input("Nottingham prognostic index", 1.0, 7.0, 5.1, step=0.1)

    c1, c2, c3 = st.columns(3)
    er_status = c1.selectbox("ER status", ref["ER Status"], index=ref["ER Status"].index("Negative"))
    pr_status = c2.selectbox("PR status", ref["PR Status"], index=ref["PR Status"].index("Negative"))
    menopausal = c3.selectbox("Menopausal state", ref["Inferred Menopausal State"],
                               index=ref["Inferred Menopausal State"].index("Post"))

    c1, c2, c3 = st.columns(3)
    chemo = c1.selectbox("Chemotherapy", ref["Chemotherapy"], index=ref["Chemotherapy"].index("Yes"))
    hormone = c2.selectbox("Hormone therapy", ref["Hormone Therapy"], index=ref["Hormone Therapy"].index("No"))
    radio = c3.selectbox("Radiotherapy", ref["Radio Therapy"], index=ref["Radio Therapy"].index("Yes"))

    surgery = st.selectbox("Type of breast surgery", ref["Type of Breast Surgery"],
                            index=ref["Type of Breast Surgery"].index("Mastectomy"))

    st.subheader("Molecular / genomic")
    mutation_count = st.slider("Mutation count", 1, 30, 6)

    with st.container():
        st.markdown('<div class="genomic-panel">', unsafe_allow_html=True)
        pam50 = st.selectbox(
            "PAM50 + claudin-low subtype", ref["Pam50 + Claudin-low subtype"],
            index=ref["Pam50 + Claudin-low subtype"].index("Basal"),
        )
        st.caption(PAM50_INFO.get(pam50, ""))
        her2_snp6 = st.selectbox(
            "HER2 status (SNP6 copy number)", ref["HER2 status measured by SNP6"],
            index=ref["HER2 status measured by SNP6"].index("Neutral"),
        )
        integrative_cluster = st.selectbox(
            "Integrative cluster", ref["Integrative Cluster"],
            index=ref["Integrative Cluster"].index("10"),
        )
        gene3_subtype = st.selectbox(
            "3-gene classifier subtype", ref["3-Gene classifier subtype"],
            index=ref["3-Gene classifier subtype"].index("ER-/HER2-"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

row = pd.DataFrame([{
    "Age at Diagnosis": age,
    "Tumor Size": tumor_size,
    "Tumor Stage": tumor_stage,
    "Neoplasm Histologic Grade": grade,
    "Lymph nodes examined positive": nodes_pos,
    "Nottingham prognostic index": npi,
    "ER Status": er_status,
    "PR Status": pr_status,
    "Chemotherapy": chemo,
    "Hormone Therapy": hormone,
    "Radio Therapy": radio,
    "Type of Breast Surgery": surgery,
    "Inferred Menopausal State": menopausal,
    "Mutation Count": mutation_count,
    "Pam50 + Claudin-low subtype": pam50,
    "HER2 status measured by SNP6": her2_snp6,
    "Integrative Cluster": integrative_cluster,
    "3-Gene classifier subtype": gene3_subtype,
}])

p_full = float(pipeline_full.predict_proba(row)[0, 1])
p_clinical = float(pipeline_clinical.predict_proba(row)[0, 1])
delta_pts = (p_full - p_clinical) * 100
label, color = risk_category(p_full)

with col_result:
    st.subheader("Estimated disease-specific mortality")
    st.markdown(
        f"""
        <div style="text-align:center; padding: 10px 0 4px 0;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:44px; font-weight:600; color:#12201c;">
                {p_full*100:.1f}%
            </span>
        </div>
        <div style="text-align:center; margin-bottom: 18px;">
            <span style="background-color:{color}1a; color:{color}; padding:4px 14px;
                         border-radius:999px; font-size:13px; font-weight:600;">
                {label} risk
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(p_full, 1.0))

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Clinical-only estimate", f"{p_clinical*100:.1f}%")
    r2.metric("Genomic adjustment", f"{delta_pts:+.1f} pts")
    r3.metric("Full model estimate", f"{p_full*100:.1f}%")

    st.caption(
        "Trained on the public METABRIC research cohort (real, de-identified "
        "patient data). Estimates disease-specific mortality, not overall "
        "survival. This is a methods demonstration, not a validated clinical "
        "tool — see the README for cohort details and limitations."
    )

st.divider()
st.caption(
    "Source: METABRIC cohort via cBioPortal. Model: logistic regression, "
    "clinical + genomic features. See project README for full methodology."
)
