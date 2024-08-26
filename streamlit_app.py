import pickle
import streamlit as st

# Load pkl file
try:
    with open('model_diabetes.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display model type
st.write(f"Model type: {type(model)}")

# If it's a numpy.ndarray, it's not a model
if isinstance(model, (list, np.ndarray)):
    st.error("Loaded object is not a valid scikit-learn model.")
    st.stop()

# Title the page
st.title("Diabetes Prediction")

# Inputs
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=10, value=1)
Glucose = st.number_input('Glucose', min_value=0, max_value=300, value=1)
Insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=1)
BMI = st.number_input('BMI', min_value=0.0, max_value=100.0, value=1.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=1.0)
Age = st.number_input('Age', min_value=0, max_value=120, value=1)

# Predict
try:
    output = model.predict([[Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    st.write("Diabetes Prediction:", round(output[0], 2))
except Exception as e:
    st.error(f"Error making prediction: {e}")
