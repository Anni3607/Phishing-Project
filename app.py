import streamlit as st
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# =====================================
# LOAD ARTIFACTS
# =====================================

model = joblib.load("model_context_aware.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("sender_scaler.pkl")
trust_dict = joblib.load("sender_trust_scores.pkl")

GLOBAL_TRUST_PRIOR = 0.5

# =====================================
# FEATURE ENGINEERING
# =====================================

def extract_structural_features(text):
    num_urls = len(re.findall(r'http[s]?://', text))
    domains = re.findall(r'http[s]?://([^/]+)/?', text)
    num_unique_domains = len(set(domains))
    has_ip_url = 1 if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', text) else 0
    suspicious_tld = 1 if any(tld in text.lower() for tld in ['.ru','.tk','.xyz','.top']) else 0
    exclamation_count = text.count("!")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text),1)
    urgent_flag = 1 if any(w in text.lower() for w in ['urgent','immediately','verify','suspend']) else 0

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
        return GLOBAL_TRUST_PRIOR, True
    sender = sender.lower().strip()
    if sender in trust_dict:
        return trust_dict[sender], False
    return GLOBAL_TRUST_PRIOR, True

# =====================================
# UI
# =====================================

st.set_page_config(page_title="Context-Aware Phishing Detector", layout="wide")

st.title("Context-Aware Phishing Detection")
st.markdown("Email Text + Sender Behavioral Modeling")

sender_input = st.text_input("Sender Email")
email_text = st.text_area("Email Content", height=200)

analyze = st.button("Analyze Email")

if analyze and email_text.strip():

    struct_features = extract_structural_features(email_text)
    trust_score, cold_start = get_trust_score(sender_input)

    text_vec = vectorizer.transform([email_text])
    numeric = np.array(struct_features + [trust_score]).reshape(1,-1)
    numeric_scaled = scaler.transform(numeric)
    final_input = hstack([text_vec, numeric_scaled])

    probability = model.predict_proba(final_input)[0][1]

    tab1, tab2, tab3 = st.tabs(["Prediction", "Behavior Analysis", "Feature Visualization"])

    # ===============================
    # TAB 1
    # ===============================

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

    # ===============================
    # TAB 2
    # ===============================

    with tab2:

        st.subheader("Sender Trust Level")
        st.progress(float(trust_score))

        if trust_score < 0.3:
            st.error("Low Trust Sender")
        elif trust_score < 0.7:
            st.warning("Moderate Trust Sender")
        else:
            st.success("High Trust Sender")

        if cold_start:
            st.info("New sender detected. Trust score assigned from global prior.")

        st.write(f"Sender Trust Score: {trust_score:.4f}")

        st.subheader("Why This Was Flagged")

        reasons = []

        if struct_features[0] > 0:
            reasons.append("Contains URLs")
        if struct_features[2] == 1:
            reasons.append("Contains IP-based URL")
        if struct_features[4] > 3:
            reasons.append("High number of exclamation marks")
        if struct_features[6] == 1:
            reasons.append("Contains urgency-related language")
        if trust_score < 0.4:
            reasons.append("Low historical sender trust")

        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("No strong structural risk indicators detected.")

    # ===============================
    # TAB 3
    # ===============================

    with tab3:

        st.subheader("Structural Feature Radar")

        labels = ["URLs","Domains","IP","TLD","Exclaim","Upper","Urgency","Trust"]

        radar_values = [
            min(struct_features[0],5),
            min(struct_features[1],5),
            struct_features[2],
            struct_features[3],
            min(struct_features[4],5),
            struct_features[5]*5,
            struct_features[6]*5,
            trust_score*5
        ]

        radar_values += radar_values[:1]
        angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
        ax.plot(angles, radar_values)
        ax.fill(angles, radar_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks([])
        ax.set_title("")

        st.pyplot(fig, use_container_width=False)

elif analyze:
    st.warning("Please enter email content.")
