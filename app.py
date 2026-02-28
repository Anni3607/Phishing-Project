import streamlit as st
import joblib
import numpy as np
import re
import plotly.graph_objects as go
from scipy.sparse import hstack
import sqlite3

# ============================
# LOAD ARTIFACTS
# ============================

model = joblib.load("model_sender_trust_unified.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("structural_scaler.pkl")

GLOBAL_TRUST_PRIOR = 0.5
ALPHA = 2  # Bayesian smoothing strength

# ============================
# SQLITE DATABASE (Dynamic Trust Storage)
# ============================

conn = sqlite3.connect("sender_reputation.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sender_reputation (
    sender TEXT PRIMARY KEY,
    legit_count INTEGER,
    phish_count INTEGER
)
""")
conn.commit()

# ============================
# FEATURE ENGINEERING
# ============================

def extract_structural_features(text):
    text = str(text)

    num_urls = len(re.findall(r'https?://\S+|www\.\S+', text))
    domains = re.findall(r'https?://([^/\s]+)', text)
    num_unique_domains = len(set(domains))
    has_ip_url = 1 if re.search(r'https?://\d+\.\d+\.\d+\.\d+', text) else 0

    suspicious_tlds = ['.ru', '.tk', '.xyz', '.top', '.click']
    suspicious_tld = 1 if any(tld in text.lower() for tld in suspicious_tlds) else 0

    exclamation_count = text.count("!")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    urgent_words = ['urgent', 'immediately', 'verify', 'suspend', 'action required']
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

# ============================
# DYNAMIC TRUST FUNCTIONS
# ============================

def get_trust_score(sender):
    if not sender:
        return GLOBAL_TRUST_PRIOR

    sender = sender.strip().lower()

    cursor.execute("SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?", (sender,))
    row = cursor.fetchone()

    if row:
        legit, phish = row
        trust = (legit + ALPHA) / (legit + phish + 2 * ALPHA)
        return trust
    else:
        return GLOBAL_TRUST_PRIOR


def update_reputation(sender, predicted_label):
    if not sender:
        return

    sender = sender.strip().lower()

    cursor.execute("SELECT legit_count, phish_count FROM sender_reputation WHERE sender=?", (sender,))
    row = cursor.fetchone()

    if row:
        legit, phish = row
    else:
        legit, phish = 0, 0

    if predicted_label == 0:
        legit += 1
    else:
        phish += 1

    cursor.execute("""
    INSERT OR REPLACE INTO sender_reputation (sender, legit_count, phish_count)
    VALUES (?, ?, ?)
    """, (sender, legit, phish))

    conn.commit()

# ============================
# UI
# ============================

st.set_page_config(page_title="Sender-Trust Phishing Detection", layout="wide")
st.title("Sender-Trust Aware Phishing Detection")
st.caption("TF-IDF + Structural Features + Dynamic Sender Behavioral Trust")

sender_input = st.text_input("Sender Email")
email_text = st.text_area("Email Content", height=220)

if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter email content.")
        st.stop()

    struct_features = extract_structural_features(email_text)
    trust_score = get_trust_score(sender_input)

    text_features = vectorizer.transform([email_text])

    struct_array = np.array(struct_features).reshape(1, -1)
    struct_scaled = scaler.transform(struct_array)

    trust_array = np.array([[trust_score]])
    numeric_features = np.hstack([struct_scaled, trust_array])

    final_features = hstack([text_features, numeric_features])

    probability = model.predict_proba(final_features)[0][1]
    prob_pct = probability * 100
    trust_pct = trust_score * 100

    # Text-only comparison
    zero_numeric = np.zeros_like(numeric_features)
    text_only_features = hstack([text_features, zero_numeric])
    text_prob_pct = model.predict_proba(text_only_features)[0][1] * 100

    # Update dynamic trust AFTER prediction
    predicted_label = 1 if probability >= 0.5 else 0
    update_reputation(sender_input, predicted_label)

    # ============================
    # TABS
    # ============================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Prediction",
        "Behavior Analysis",
        "Feature Visualization",
        "Feature Contribution",
        "Model Comparison",
        "Probability Decomposition",
        "Top Word Signals"
    ])

    # TAB 1
    with tab1:
        st.subheader("Risk Level")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {
                    'color': "red" if prob_pct > 70
                    else "orange" if prob_pct > 40
                    else "green"
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

        if prob_pct > 70:
            st.error("High Risk: Phishing Likely")
        elif prob_pct > 40:
            st.warning("Medium Risk: Suspicious")
        else:
            st.success("Low Risk: Likely Legitimate")

        st.write(f"Sender Trust Score (Dynamic): {trust_pct:.1f}%")

    # TAB 2
    with tab2:
        st.subheader("Sender Trust (Live Reputation)")
        st.progress(float(trust_score))
        st.write(f"Trust Score: {trust_pct:.1f}%")

    # TAB 3
    with tab3:
        labels = ["URLs","Domains","IP","TLD",
                  "Exclaim","Uppercase","Urgency","Trust"]

        values = [
            min(struct_features[0],5),
            min(struct_features[1],5),
            struct_features[2]*5,
            struct_features[3]*5,
            min(struct_features[4],5),
            struct_features[5]*5,
            struct_features[6]*5,
            trust_score*5
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), height=450)
        st.plotly_chart(fig, width="stretch")

    # TAB 4
    with tab4:
        if hasattr(model,"coef_"):
            coefficients = model.coef_[0]
            text_dim = text_features.shape[1]
            numeric_coefs = coefficients[text_dim:]
            feature_names = ["URLs","Domains","IP URL","Suspicious TLD",
                             "Exclamation","Uppercase","Urgency","Trust"]
            contributions = numeric_features.flatten()*numeric_coefs

            fig = go.Figure(go.Bar(x=contributions, y=feature_names, orientation='h'))
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

    # TAB 5
    with tab5:
        col1,col2 = st.columns(2)
        col1.metric("Text-only Probability", f"{text_prob_pct:.1f}%")
        col2.metric("Trust-Aware Probability", f"{prob_pct:.1f}%")

    # TAB 6
    with tab6:
        if hasattr(model,"coef_"):
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            text_dim = text_features.shape[1]

            text_contribution = (text_features @ coefficients[:text_dim])[0]
            numeric_contribution = np.dot(numeric_features.flatten(), coefficients[text_dim:])
            total_logit = intercept + text_contribution + numeric_contribution

            st.write("Intercept:", round(float(intercept),4))
            st.write("Text Contribution:", round(float(text_contribution),4))
            st.write("Numeric Contribution:", round(float(numeric_contribution),4))
            st.write("Final Logit:", round(float(total_logit),4))
            st.write("Final Probability:", f"{prob_pct:.2f}%")

    # TAB 7
    with tab7:
        if hasattr(model,"coef_"):
            coefficients = model.coef_[0]
            text_dim = text_features.shape[1]
            feature_names = vectorizer.get_feature_names_out()
            text_vector = text_features.toarray().flatten()
            word_contributions = text_vector * coefficients[:text_dim]

            top_pos = np.argsort(word_contributions)[-10:]
            top_neg = np.argsort(word_contributions)[:10]

            st.subheader("Top Phishing Indicators")
            for idx in reversed(top_pos):
                if word_contributions[idx] > 0:
                    st.write(feature_names[idx], round(word_contributions[idx],4))

            st.subheader("Top Legitimate Indicators")
            for idx in top_neg:
                if word_contributions[idx] < 0:
                    st.write(feature_names[idx], round(word_contributions[idx],4))
