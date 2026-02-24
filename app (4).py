import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# =====================================
# LOAD ARTIFACTS
# =====================================

MODEL_PATH = "phishing_artifacts/model_context_aware.pkl"
VECTORIZER_PATH = "phishing_artifacts/tfidf_vectorizer.pkl"
SCALER_PATH = "phishing_artifacts/sender_scaler.pkl"
TRUST_PATH = "phishing_artifacts/sender_trust_scores.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)
trust_dict = joblib.load(TRUST_PATH)

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
        return 0.5
    return trust_dict.get(sender.lower(), 0.5)

# =====================================
# STREAMLIT UI
# =====================================

st.set_page_config(page_title="Context-Aware Phishing Detector")

st.title("Context-Aware Phishing Detection")
st.write("Text + Sender Behavior Based Detection")

sender_input = st.text_input("Sender Email")
email_text = st.text_area("Email Content")

if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter email content.")
    else:
        # Text features
        text_features = vectorizer.transform([email_text])

        # Structural features
        struct_features = extract_structural_features(email_text)

        # Trust score
        trust_score = get_trust_score(sender_input)

        numeric_features = np.array(struct_features + [trust_score]).reshape(1, -1)
        numeric_scaled = scaler.transform(numeric_features)

        # Combine
        final_features = hstack([text_features, numeric_scaled])

        # Predict
        probability = model.predict_proba(final_features)[0][1]
        prediction = model.predict(final_features)[0]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"Phishing Detected")
        else:
            st.success("Legitimate Email")

        st.write(f"Phishing Probability: {probability:.4f}")
        st.write(f"Sender Trust Score: {trust_score:.4f}")