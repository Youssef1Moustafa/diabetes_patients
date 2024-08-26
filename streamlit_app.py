import streamlit as st
import pickle
import numpy as np

# Load the model from the file
with open('model_svm.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the page
st.title("Diabetes Prediction")

# Input fields
Pregnancies = st.number_input('Pregnancies', min_value=0.0, max_value=7.0, value=1.0)
Glucose = st.number_input('Glucose', min_value=0.0, max_value=7.0, value=1.0)
Insulin = st.number_input('Insulin', min_value=0.0, max_value=7.0, value=1.0)
BMI = st.number_input('BMI', min_value=0.0, max_value=7.0, value=1.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=7.0, value=1.0)
Age = st.number_input('Age', min_value=0.0, max_value=7.0, value=1.0)

# Create feature array for prediction
input_data = np.array([[Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Make prediction
if st.button('Predict'):
    try:
        output = model.predict(input_data)
        st.write("Prediction:", "Diabetic" if output[0] == 1 else "Not Diabetic")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

