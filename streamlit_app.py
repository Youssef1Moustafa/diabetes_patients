import os
import streamlit as st
import pickle
import numpy as np

# Check if the file exists
file_path = 'model_features.pkl'
if os.path.exists(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
else:
    st.error(f"File {file_path} not found. Please ensure the file is in the correct location.")
    st.stop()

# Extract the model and feature names
model = model_data['model']
features = model_data['features']

# Title of the page
st.title("Diabetes Prediction")

# Create input fields dynamically based on feature names
input_data = []
for feature in features:
    value = st.number_input(feature, min_value=0.0, max_value=200.0, value=1.0)  # Adjust min/max values as needed
    input_data.append(value)

# Convert input_data to a numpy array with the shape (1, -1) for prediction
input_data = np.array([input_data])

# Make prediction
if st.button('Predict'):
    try:
        output = model.predict(input_data)
        st.write("Prediction:", "Diabetic" if output[0] == 1 else "Not Diabetic")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
