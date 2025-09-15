import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import os
import time
from streamlit_lottie import st_lottie

# Load Lottie animation JSON files locally
def load_lottie_local(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return None

# Load animations (place these files in the same folder as this script)
lottie_wine = load_lottie_local("lf20_gfqwscjp.json")       # Wine glass animation
lottie_success = load_lottie_local("lf20_jbrw3hcz.json")    # Success animation
lottie_error = load_lottie_local("lf20_touohxv0.json")      # Error animation

@st.cache_resource(show_spinner=False)
def load_ann_model_and_scaler():
    try:
        model = tf.keras.models.load_model("ann_wine_model.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def user_input_features():
    st.sidebar.header("Input Wine Chemical Properties")
    data = {
        'fixed acidity': st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.9, 0.1),
        'volatile acidity': st.sidebar.slider('Volatile Acidity', 0.1, 1.5, 0.32, 0.01),
        'citric acid': st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.51, 0.01),
        'residual sugar': st.sidebar.slider('Residual Sugar', 0.5, 15.0, 1.8, 0.1),
        'chlorides': st.sidebar.slider('Chlorides', 0.01, 0.2, 0.055, 0.001),
        'free sulfur dioxide': st.sidebar.slider('Free Sulfur Dioxide', 1, 70, 23, 1),
        'total sulfur dioxide': st.sidebar.slider('Total Sulfur Dioxide', 6, 200, 49, 1),
        'density': st.sidebar.slider('Density', 0.9900, 1.0050, 0.9956, 0.0001),
        'pH': st.sidebar.slider('pH', 2.8, 4.0, 3.30, 0.01),
        'sulphates': st.sidebar.slider('Sulphates', 0.3, 1.5, 0.73, 0.01),
        'alcohol': st.sidebar.slider('Alcohol (%)', 8.0, 15.0, 12.1, 0.1),
    }
    return pd.DataFrame(data, index=[0])

def main():
    st.set_page_config(page_title="🍷 Wine Quality Prediction (ANN)", layout="wide", page_icon="🍷")

    # Header section with animation
    st.markdown("<h1 style='text-align:center;'>🍷 Wine Quality Prediction App (ANN)</h1>", unsafe_allow_html=True)
    if lottie_wine:
        st_lottie(lottie_wine, height=150)

    st.markdown("---")
    st.write("Enter the wine chemical properties in the sidebar and click **Predict** to see if the wine quality is predicted to be Good or Poor.")

    # Get user inputs
    input_df = user_input_features()

    st.subheader("Input parameters")
    st.write(input_df)

    # Load ANN model and scaler
    model, scaler = load_ann_model_and_scaler()
    if model is None or scaler is None:
        st.stop()

    # Ensure columns order match training data
    expected_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]
    input_df = input_df[expected_cols]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict button
    if st.button("Predict Wine Quality 🍇"):
        with st.spinner("Predicting..."):
            time.sleep(1)
            prob = model.predict(input_scaled)[0][0]
            prediction = 1 if prob >= 0.5 else 0
            confidence = prob if prediction == 1 else 1 - prob

        if prediction == 1:
            st.success(f"🎉 This wine is predicted to be of **GOOD** quality! (Confidence: {confidence:.2%})")
            if lottie_success:
                st_lottie(lottie_success, height=180)
        else:
            st.error(f"🚫 This wine is predicted to be of **POOR** quality. (Confidence: {confidence:.2%})")
            if lottie_error:
                st_lottie(lottie_error, height=180)

    st.markdown("---")

    st.subheader("Understanding Wine Quality Factors")
    st.markdown("""
    - **Alcohol:** Higher content often means better quality.
    - **Volatile Acidity:** Lower levels indicate better wine.
    - **Sulphates:** Natural preservative, affects freshness.
    - **Chlorides:** Lower salt for better taste.
    - **Citric Acid:** Adds crispness and complexity.
    """)

    st.caption("Developed with ❤️ using Streamlit & TensorFlow")

if __name__ == "__main__":
    main()
