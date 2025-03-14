import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define model path
MODEL_PATH = r"C:\Users\Asus\PycharmProjects\pythonProject40\lstm_model.h5"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model first.")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

# Streamlit App Title
st.title("Student Depression Analysis Dashboard")
st.write("Upload your dataset, visualize trends, and predict depression risk.")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna()

    # Data Preprocessing
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No": 0, "Yes": 1})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No": 0, "Yes": 1})
    df["Depression"] = df["Depression"].astype(int)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Interactive Data Visualization
    st.subheader("Depression Distribution")
    fig = px.histogram(df, x="Depression", title="Depression Cases in Students", color="Depression")
    st.plotly_chart(fig)

    st.subheader("CGPA vs Depression")
    fig = px.scatter(df, x="CGPA", y="Depression", color="Depression", title="CGPA vs Depression")
    st.plotly_chart(fig)

# Real-time Prediction
st.sidebar.subheader("Predict Depression Risk")
inputs = {}
features = ["Gender", "Age", "Academic Pressure", "Work Pressure", "CGPA",
            "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
            "Financial Stress", "Family History of Mental Illness",
            "Have you ever had suicidal thoughts ?"]

for feature in features:
    inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

if st.sidebar.button("Predict Depression"):
    input_data = np.array(list(inputs.values())).reshape(1, 1, -1)  # Reshape for LSTM
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data.reshape(1, -1)).reshape(1, 1, -1)
    prediction = model.predict(input_data_scaled)
    result = "High Risk of Depression" if prediction > 0.5 else "Low Risk of Depression"
    st.sidebar.subheader(f"Prediction: {result}")
