import streamlit as st
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# =====================================
# LOAD ARTIFACTS
# =====================================

MODEL_PATH = "model_context_aware.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
SCALER_PATH = "sender_scaler.pkl"
TRUST_PATH = "sender_trust_scores.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)
trust_dict = joblib.load(TRUST_PATH)

GLOBAL_TRUST_PRIOR = 0.5

# =====================================
# FEATURE ENGINEERING
# =====================================

def extract_structural_features(text: str):
    num_urls = len(re.findall(r'http[s]?://', text))
    domains = re.findall(r'http[s]?://([^/]+)/?', text)
    num_unique_domains = len(set(domains))
    has_ip_url = 1 if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', text) else 0

    suspicious_tlds = ['.ru', '.tk', '.xyz', '.top']
    suspicious_tld = 1 if any(tld in text.lower() for tld in suspicious_tlds) else 0

    exclamation_count = text.count("!")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    urgent_words = ['urgent', 'immediately', 'verify', 'suspend']
    urgent_flag = 1 if any(word in text.lower() for word in urgent_words) else 0

    return [
        num_urls,
        num_unique_domains,
        has_ip_url,
        suspicious_tld,
        exclamation_count,
        uppercase_ratio,
        urgent_flag
    ]

def get_trust_score(sender: str):
    if not sender:
        return GLOBAL_TRUST_PRIOR
    sender = sender.lower().strip()
    return trust_dict.get(sender, GLOBAL_TRUST_PRIOR)

# =====================================
# STREAMLIT CONFIG
# =====================================

st.set_page_config(page_title="Context-Aware Phishing Detector", layout="wide")

st.title("Context-Aware Phishing Detection")
st.markdown("Email Text + Sender Behavioral Modeling")

sender_input = st.text_input("Sender Email")
email_text = st.text_area("Email Content", height=200)

if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter email content.")
        st.stop()

    # Extract features
    struct_features = extract_structural_features(email_text)
    trust_score = get_trust_score(sender_input)

    text_features = vectorizer.transform([email_text])
    numeric_features = np.array(struct_features + [trust_score]).reshape(1, -1)
    numeric_scaled = scaler.transform(numeric_features)

    final_features = hstack([text_features, numeric_scaled])

    probability_context = model.predict_proba(final_features)[0][1]
    prediction = model.predict(final_features)[0]

    # TEXT-ONLY MODEL (simulate by zeroing numeric features)
    zero_numeric = np.zeros_like(numeric_scaled)
    text_only_features = hstack([text_features, zero_numeric])
    probability_text_only = model.predict_proba(text_only_features)[0][1]

    # Convert to %
    prob_context_pct = probability_context * 100
    prob_text_pct = probability_text_only * 100
    trust_pct = trust_score * 100

    tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Contribution", "Comparison"])

    # =====================================================
    # TAB 1 — PREDICTION
    # =====================================================

    with tab1:

        st.subheader("Risk Score")

        col1, col2 = st.columns([3, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh(["Risk"], [prob_context_pct])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_yticks([])
            ax.set_facecolor("none")
            fig.patch.set_facecolor("none")
            st.pyplot(fig)

        with col2:
            st.metric("Phishing Probability", f"{prob_context_pct:.1f}%")

        if probability_context > 0.7:
            st.error("High Risk: Phishing Likely")
        elif probability_context > 0.4:
            st.warning("Medium Risk: Suspicious")
        else:
            st.success("Low Risk: Likely Legitimate")

        st.write(f"Sender Trust Score: {trust_pct:.1f}%")

    # =====================================================
    # TAB 2 — FEATURE CONTRIBUTION
    # =====================================================

    with tab2:

        st.subheader("Feature Contribution (Numeric + Trust)")

        if hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            text_dim = text_features.shape[1]

            numeric_coefs = coefficients[text_dim:]

            feature_names = [
                "URLs",
                "Domains",
                "IP URL",
                "Suspicious TLD",
                "Exclamation",
                "Uppercase Ratio",
                "Urgency",
                "Trust"
            ]

            contributions = numeric_scaled.flatten() * numeric_coefs

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(feature_names, contributions)
            ax.set_title("Feature Impact on Log-Odds")
            ax.set_facecolor("none")
            fig.patch.set_facecolor("none")
            st.pyplot(fig)
        else:
            st.info("Feature contribution only supported for linear models.")

    # =====================================================
    # TAB 3 — TEXT vs CONTEXT COMPARISON
    # =====================================================

    with tab3:

        st.subheader("Text-only vs Context-aware")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Text-only Probability", f"{prob_text_pct:.1f}%")

        with col2:
            st.metric("Context-aware Probability", f"{prob_context_pct:.1f}%")

        delta = prob_context_pct - prob_text_pct

        if delta > 0:
            st.write(f"Behavioral context increased risk by {delta:.1f}%")
        elif delta < 0:
            st.write(f"Behavioral context reduced risk by {abs(delta):.1f}%")
        else:
            st.write("Behavioral context had no impact.")
