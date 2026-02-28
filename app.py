import streamlit as st
import joblib
import numpy as np
import re
import plotly.graph_objects as go
from scipy.sparse import hstack

# ============================
# LOAD ARTIFACTS
# ============================

model = joblib.load("model_context_aware.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("sender_scaler.pkl")
trust_dict = joblib.load("sender_trust_scores.pkl")

GLOBAL_TRUST_PRIOR = 0.5

# ============================
# FEATURE ENGINEERING
# ============================

def extract_structural_features(text):
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

def get_trust_score(sender):
    if not sender:
        return GLOBAL_TRUST_PRIOR
    return trust_dict.get(sender.lower().strip(), GLOBAL_TRUST_PRIOR)

# ============================
# UI CONFIG
# ============================

st.set_page_config(page_title="Context-Aware Phishing Detection", layout="wide")
st.title("Context-Aware Phishing Detection")
st.caption("Email Text + Sender Behavioral Modeling")

sender_input = st.text_input("Sender Email")
email_text = st.text_area("Email Content", height=200)

if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter email content.")
        st.stop()

    # Feature processing
    struct_features = extract_structural_features(email_text)
    trust_score = get_trust_score(sender_input)

    text_features = vectorizer.transform([email_text])
    numeric_features = np.array(struct_features + [trust_score]).reshape(1, -1)
    numeric_scaled = scaler.transform(numeric_features)
    final_features = hstack([text_features, numeric_scaled])

    probability = model.predict_proba(final_features)[0][1]
    prediction = model.predict(final_features)[0]

    # Text-only comparison
    zero_numeric = np.zeros_like(numeric_scaled)
    text_only_features = hstack([text_features, zero_numeric])
    probability_text_only = model.predict_proba(text_only_features)[0][1]

    prob_pct = probability * 100
    text_prob_pct = probability_text_only * 100
    trust_pct = trust_score * 100

    # ============================
    # TABS
    # ============================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Prediction",
        "Behavior Analysis",
        "Feature Visualization",
        "Feature Contribution",
        "Model Comparison"
    ])

    # ==================================
    # TAB 1 — PREDICTION
    # ==================================

    with tab1:

        st.subheader("Risk Level")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob_pct > 70 else "orange" if prob_pct > 40 else "green"},
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        if prob_pct > 70:
            st.error("High Risk: Phishing Likely")
        elif prob_pct > 40:
            st.warning("Medium Risk: Suspicious")
        else:
            st.success("Low Risk: Likely Legitimate")

        st.write(f"Sender Trust Score: {trust_pct:.1f}%")

    # ==================================
    # TAB 2 — BEHAVIOR ANALYSIS
    # ==================================

    with tab2:

        st.subheader("Sender Trust")

        st.progress(trust_score)
        st.write(f"Trust Score: {trust_pct:.1f}%")

        st.subheader("Why This Was Flagged")

        reasons = []
        if struct_features[0] > 0:
            reasons.append("Contains URLs")
        if struct_features[2] == 1:
            reasons.append("Contains IP-based URL")
        if struct_features[3] == 1:
            reasons.append("Suspicious Top-Level Domain")
        if struct_features[6] == 1:
            reasons.append("Urgency-related Language")

        if trust_score < 0.4:
            reasons.append("Low Sender Trust Score")

        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("No strong structural red flags detected.")

    # ==================================
    # TAB 3 — FEATURE VISUALIZATION
    # ==================================

    with tab3:

        labels = ["URLs", "Domains", "IP", "TLD",
                  "Exclaim", "Uppercase", "Urgency", "Trust"]

        values = [
            min(struct_features[0], 5),
            min(struct_features[1], 5),
            struct_features[2] * 5,
            struct_features[3] * 5,
            min(struct_features[4], 5),
            struct_features[5] * 5,
            struct_features[6] * 5,
            trust_score * 5
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================================
    # TAB 4 — FEATURE CONTRIBUTION
    # ==================================

    with tab4:

        if hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            text_dim = text_features.shape[1]
            numeric_coefs = coefficients[text_dim:]

            feature_names = [
                "URLs", "Domains", "IP URL", "Suspicious TLD",
                "Exclamation", "Uppercase", "Urgency", "Trust"
            ]

            contributions = numeric_scaled.flatten() * numeric_coefs

            fig = go.Figure(go.Bar(
                x=contributions,
                y=feature_names,
                orientation='h'
            ))

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature contribution available only for linear models.")

    # ==================================
    # TAB 5 — MODEL COMPARISON
    # ==================================

    with tab5:

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Text-only Probability", f"{text_prob_pct:.1f}%")

        with col2:
            st.metric("Context-aware Probability", f"{prob_pct:.1f}%")

        delta = prob_pct - text_prob_pct

        if delta > 0:
            st.write(f"Behavioral context increased risk by {delta:.1f}%")
        elif delta < 0:
            st.write(f"Behavioral context reduced risk by {abs(delta):.1f}%")
        else:
            st.write("Behavioral context had no measurable impact.")
