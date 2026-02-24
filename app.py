import streamlit as st
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import os

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

    return {
        "num_urls": num_urls,
        "num_unique_domains": num_unique_domains,
        "has_ip_url": has_ip_url,
        "suspicious_tld": suspicious_tld,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": uppercase_ratio,
        "urgent_flag": urgent_flag
    }

def get_trust_score(sender: str):
    if not sender:
        return GLOBAL_TRUST_PRIOR, True
    sender = sender.lower().strip()
    if sender in trust_dict:
        return trust_dict[sender], False
    return GLOBAL_TRUST_PRIOR, True

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

    # -------------------------
    # Feature Extraction
    # -------------------------

    struct_dict = extract_structural_features(email_text)
    trust_score, is_cold_start = get_trust_score(sender_input)

    text_features = vectorizer.transform([email_text])

    numeric_features = np.array(list(struct_dict.values()) + [trust_score]).reshape(1, -1)
    numeric_scaled = scaler.transform(numeric_features)

    final_features = hstack([text_features, numeric_scaled])

    probability = model.predict_proba(final_features)[0][1]
    prediction = model.predict(final_features)[0]

    # -------------------------
    # TABS
    # -------------------------

    tab1, tab2, tab3 = st.tabs(["Prediction", "Behavior Analysis", "Feature Visualization"])

    # =====================================================
    # TAB 1 — PREDICTION
    # =====================================================

    with tab1:

        st.subheader("Risk Level")

        st.progress(float(probability))

        if probability > 0.7:
            st.error("High Risk: Phishing Likely")
        elif probability > 0.4:
            st.warning("Medium Risk: Suspicious Email")
        else:
            st.success("Low Risk: Likely Legitimate")

        st.subheader("Prediction Distribution")

        st.bar_chart({
            "Legitimate": [1 - probability],
            "Phishing": [probability]
        })

        st.write(f"Phishing Probability: {probability:.4f}")

    # =====================================================
    # TAB 2 — BEHAVIOR ANALYSIS
    # =====================================================

    with tab2:

        st.subheader("Sender Trust Level")

        st.progress(float(trust_score))

        if trust_score < 0.3:
            st.error("Low Trust Sender")
        elif trust_score < 0.7:
            st.warning("Moderate Trust Sender")
        else:
            st.success("High Trust Sender")

        if is_cold_start:
            st.info("New sender detected. Trust score assigned from global prior.")

        st.write(f"Sender Trust Score: {trust_score:.4f}")

        st.subheader("Why This Was Flagged")

        reasons = []

        if struct_dict["num_urls"] > 0:
            reasons.append("Contains URLs")

        if struct_dict["has_ip_url"] == 1:
            reasons.append("Contains IP-based URL")

        if struct_dict["suspicious_tld"] == 1:
            reasons.append("Contains suspicious top-level domain")

        if struct_dict["exclamation_count"] > 3:
            reasons.append("High number of exclamation marks")

        if struct_dict["urgent_flag"] == 1:
            reasons.append("Contains urgency-related language")

        if trust_score < 0.4:
            reasons.append("Sender has low historical trust score")

        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("No strong structural risk indicators detected.")

    # =====================================================
    # TAB 3 — FEATURE VISUALIZATION
    # =====================================================

   with tab3:

    st.subheader("Structural Feature Radar")

    labels = ["URLs", "Domains", "IP URL", "TLD",
              "Exclaim", "Uppercase", "Urgency", "Trust"]

    values = [
        min(struct_dict["num_urls"], 5),
        min(struct_dict["num_unique_domains"], 5),
        struct_dict["has_ip_url"],
        struct_dict["suspicious_tld"],
        min(struct_dict["exclamation_count"], 5),
        struct_dict["uppercase_ratio"] * 5,
        struct_dict["urgent_flag"] * 5,
        trust_score * 5
    ]

    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_yticks([])
    ax.set_title("")

    st.pyplot(fig, use_container_width=False)

