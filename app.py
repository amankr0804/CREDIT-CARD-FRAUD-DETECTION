import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#101d35,#183b59); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; color:white;}
.hero p {color:#d4e3f4; margin:.4rem 0 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ Credit Card Fraud Detection</h1><p>Transaction risk assessment using your trained machine-learning models</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    assets = {}
    files = {
        "XGBoost": "xgboost.pkl",
        "Random Forest": "random_forest.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Isolation Forest": "isolation_forest.pkl",
        "Amount scaler": "scaler.pkl",
        "Time scaler": "time_scaler.pkl",
    }
    for key, filename in files.items():
        path = MODEL_DIR / filename
        if path.exists():
            assets[key] = joblib.load(path)
    return assets

assets = load_assets()
models = [name for name in ["XGBoost", "Random Forest", "Logistic Regression", "Isolation Forest"] if name in assets]
if not models:
    st.error("No trained model files found. Put your .pkl files in a folder named 'models' beside app.py.")
    st.code("models/xgboost.pkl\nmodels/random_forest.pkl\nmodels/logistic_regression.pkl\nmodels/isolation_forest.pkl\nmodels/scaler.pkl")
    st.stop()

st.sidebar.header("Dashboard")
mode = st.sidebar.radio("Choose input", ["Single transaction", "Batch CSV"])
model_name = st.sidebar.selectbox("Model", models, index=models.index("XGBoost") if "XGBoost" in models else 0)
st.sidebar.caption("Supervised models use a 0.50 probability threshold. Isolation Forest returns an anomaly flag.")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def prep(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = df[FEATURES].copy().astype(float)
    if "Amount scaler" in assets:
        X["Amount"] = assets["Amount scaler"].transform(X[["Amount"]]).ravel()
    if "Time scaler" in assets:
        X["Time"] = assets["Time scaler"].transform(X[["Time"]]).ravel()
    return X

def risk_label(p):
    if p < .30: return "Low"
    if p < .60: return "Medium"
    if p < .85: return "High"
    return "Critical"

def score_frame(X, model_key):
    model = assets[model_key]
    if model_key == "Isolation Forest":
        raw = -model.score_samples(X)
        lo, hi = getattr(model, "_fraud_score_min", raw.min()), getattr(model, "_fraud_score_max", raw.max())
        score = np.clip((raw-lo)/(hi-lo), 0, 1) if hi > lo else np.zeros(len(raw))
        pred = (model.predict(X) == -1).astype(int)
        return score, pred
    score = model.predict_proba(X)[:, 1]
    return score, (score >= .5).astype(int)

if mode == "Single transaction":
    st.subheader("Enter transaction details")
    st.info("Enter Time, V1–V28 and Amount. For exact parity with training, use the same preprocessing as your notebook.")
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        vals = {}
        vals["Time"] = c1.number_input("Time", value=0.0, help="Seconds elapsed from the first transaction in the dataset.")
        vals["Amount"] = c2.number_input("Amount", min_value=0.0, value=100.0)
        st.markdown("**Anonymized PCA features (V1–V28)**")
        cols = st.columns(4)
        for i in range(1, 29):
            vals[f"V{i}"] = cols[(i-1)%4].number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}")
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)
    if submitted:
        try:
            X = prep(pd.DataFrame([vals]))
            score, pred = score_frame(X, model_name)
            if model_name == "Isolation Forest":
                st.metric("Anomaly flag", "Potential anomaly" if pred[0] else "No anomaly flag")
                st.metric("Normalized anomaly score", f"{score[0]:.1%}")
                st.caption("This score is a normalized anomaly score, not a calibrated fraud probability.")
            else:
                st.metric("Fraud probability", f"{score[0]:.2%}")
                st.metric("Risk category", risk_label(float(score[0])))
                st.metric("Model decision", "Potential fraud" if pred[0] else "Likely legitimate")
            st.warning("This is a model prediction for review, not a definitive determination of fraud.")
        except Exception as e:
            st.error(str(e))
else:
    st.subheader("Batch transaction analysis")
    uploaded = st.file_uploader("Upload CSV containing Time, V1–V28, Amount", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.write("Preview", raw.head())
            X = prep(raw)
            if st.button("Analyze batch", type="primary"):
                with st.spinner("Scoring transactions..."):
                    score, pred = score_frame(X, model_name)
                result = raw.copy()
                result["fraud_score"] = score
                result["prediction"] = np.where(pred == 1, "Potential fraud/anomaly", "Likely legitimate")
                if model_name != "Isolation Forest":
                    result["risk_level"] = [risk_label(float(x)) for x in score]
                a,b,c = st.columns(3)
                a.metric("Transactions", len(result))
                b.metric("Flagged", int(pred.sum()))
                c.metric("Flag rate", f"{pred.mean():.2%}" if len(pred) else "—")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"), "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("Models: Logistic Regression, Random Forest, XGBoost and Isolation Forest | For educational/demo use; validate thresholds and preprocessing before operational use.")
if "Time scaler" not in assets:
    st.warning("Time scaler file not found. Your notebook scales Time during training but does not save that fitted scaler. Run the small notebook fix below and save models/time_scaler.pkl; otherwise Time is passed through unchanged.")
