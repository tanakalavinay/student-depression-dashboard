import streamlit as slt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and clean the dataset
file_path = r"C:\Users\Asus\Downloads\Student Depression Dataset.csv"  # Update if needed
df = pd.read_csv(file_path).dropna()

# Convert categorical values to numerical
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No": 0, "Yes": 1})
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No": 0, "Yes": 1})
df["Depression"] = df["Depression"].astype(int)

# Feature selection
features = ["Gender", "Age", "Academic Pressure", "Work Pressure", "CGPA",
            "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
            "Financial Stress", "Family History of Mental Illness",
            "Have you ever had suicidal thoughts ?"]
X = df[features]
y = df["Depression"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for deep learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Deep Learning Model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Deep Learning Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
fig = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"),
                x=["No Depression", "Depression"], y=["No Depression", "Depression"],
                color_continuous_scale="Blues")
fig.update_layout(title="Confusion Matrix - Deep Learning Model")
fig.show()

# Feature Importance (from trained weights)
importance = abs(model.layers[0].get_weights()[0]).sum(axis=1)
feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)

# Interactive Feature Importance Plot
fig = px.bar(feature_importance, x=feature_importance.values, y=feature_importance.index, orientation="h",
             labels={"x": "Importance Score", "y": "Feature"}, title="Feature Importance in Predicting Depression")
fig.show()

# Interactive Depression Distribution by Gender
fig = px.histogram(df, x="Gender", color="Depression", barmode="group",
                   labels={"Gender": "Gender (0=Male, 1=Female)", "count": "Number of Students"},
                   title="Depression Cases by Gender")
fig.show()

# Interactive Scatter Plot - CGPA vs Depression
fig = px.scatter(df, x="CGPA", y="Depression", color="Depression",
                 labels={"CGPA": "CGPA Score", "Depression": "Depression (0=No, 1=Yes)"},
                 title="CGPA vs Depression")
fig.show()
