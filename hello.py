import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Churn Predictor")

# --- Load Model (Safe Inference Mode) ---
try:
    model = tf.keras.models.load_model('model.h5', compile=False)
except Exception as e:
    st.error(f"❌ Failed to load model. Error: {e}")
    st.stop()

# --- Load Encoders and Scaler ---
try:
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading preprocessing files. Error: {e}")
    st.stop()

# --- UI Title ---
st.title("Customer Churn Prediction")

# --- User Inputs ---
try:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
except Exception as e:
    st.error(f"❌ Error reading geography categories from encoder: {e}")
    st.stop()

gender = st.selectbox("Gender", ["Male", "Female"])
label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(["Male", "Female"])
encoded_gender = label_encoder_gender.transform([gender])[0]

age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has CreditCard", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")

# --- Construct Input DataFrame ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [encoded_gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# --- One-Hot Encode Geography ---
try:
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())
except Exception as e:
    st.error(f"❌ One-hot encoding failed: {e}")
    st.stop()

# --- Combine All Features ---
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# --- Match Feature Order for Scaler ---
try:
    input_data = input_data[scaler.feature_names_in_]
except Exception as e:
    st.error("❌ Feature mismatch with scaler. Check that input columns match those used during training.")
    st.write("Expected:", scaler.feature_names_in_)
    st.write("Got:", input_data.columns.tolist())
    st.stop()

# --- Scale Input ---
input_data_scaled = scaler.transform(input_data)

# --- Make Prediction ---
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# --- Output Result ---
st.subheader(f"🔍 Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.7:
    st.warning("⚠️ The customer is highly likely to churn.")
elif prediction_proba > 0.4:
    st.info("ℹ️ The customer has a moderate chance of churning.")
else:
    st.success("✅ The customer is unlikely to churn.")
