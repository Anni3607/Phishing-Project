Context-Aware Phishing Detection using Machine Learning and Sender Behavior Analysis

This project implements a context-aware phishing detection system that combines email content analysis with sender behavioral profiling. Unlike traditional text-only phishing detectors, this system integrates linguistic features, structural indicators, and sender trust history to produce a more informed and explainable prediction.

Overview

Phishing emails attempt to trick users into revealing sensitive information such as passwords, financial details, or confidential data. Modern phishing attacks often use well-written and professional language, making text-only detection models less reliable. Many systems focus purely on email content and ignore who is sending the email and how that sender has behaved historically.

This project addresses that gap by incorporating both content-based features and sender behavior into a unified classification framework.

Core Idea

The system operates on two main pillars:

What the email says

Who is sending it and how they have behaved before

Email content is processed using a TF-IDF vectorizer to convert text into numerical features. Structural characteristics such as number of URLs, presence of IP-based links, suspicious top-level domains, exclamation frequency, uppercase ratio, and urgency-related language are extracted. Additionally, a dynamic sender trust score is computed based on historical behavior. These features are combined and fed into a trained machine learning model to classify the email as phishing or legitimate.

System Architecture

The pipeline follows these steps:

Text Vectorization
The email body is transformed using a trained TF-IDF vectorizer.

Structural Feature Extraction
The system extracts measurable risk indicators from the raw email content, including:

Number of URLs

Number of unique domains

Presence of IP-based URLs

Suspicious top-level domains

Exclamation mark frequency

Uppercase character ratio

Urgency-related keywords

Sender Trust Modeling
A sender trust score is retrieved from historical statistics. If the sender has no history, a global prior is assigned. This models real-world cold-start scenarios.

Feature Fusion
Text features and scaled numeric features are combined into a single feature vector.

Classification
A trained machine learning model outputs the phishing probability and final prediction.

Explainability
The Streamlit interface displays:

Risk level

Phishing probability

Sender trust level

Reasons for flagging

Structural feature radar visualization

Technologies Used

Python

Streamlit

Scikit-learn

NumPy

SciPy

Matplotlib

Joblib

Project Structure

app.py
Main Streamlit application

model_context_aware.pkl
Trained classification model

tfidf_vectorizer.pkl
Trained TF-IDF vectorizer

sender_scaler.pkl
Feature scaler for numeric attributes

sender_trust_scores.pkl
Dictionary containing sender trust scores

requirements.txt
Python dependencies

How to Run Locally

Clone the repository

Install dependencies
pip install -r requirements.txt

Run the application
streamlit run app.py

The app will open in your browser and allow interactive email testing.

Deployment

The application is deployable on Streamlit Cloud. All required model artifacts are included in the repository. Ensure that the .pkl files remain in the root directory if using the current configuration.

Key Features

Context-aware classification

Behavioral sender profiling

Cold-start handling

Structural risk analysis

Visual explanation dashboard

Real-time probability scoring

Limitations

Trust scores depend on available historical data

No live email server integration

Model performance depends on training dataset quality

Does not detect zero-day phishing strategies outside learned patterns

Future Improvements

Live sender reputation APIs

Advanced explainability using SHAP

Adaptive threshold tuning

Real-time monitoring integration

Enterprise-level logging and audit trails

Conclusion

This project demonstrates how integrating sender behavior with linguistic and structural analysis can improve phishing detection beyond traditional text-only approaches. It provides a practical and explainable implementation of context-aware email security using machine learning.
