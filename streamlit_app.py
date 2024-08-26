import streamlit as st
import pickle


# load pkl file
with open('model_diabetes.pkl', 'rb') as file:
    model = pickle.load(file)

#title the page
st.title("diabetes patient")

#set image

#inputs
Pregnancies = st.number_input('Pregnancies' , min_value=0.0 , max_value=10.0,value=1.0)
Glucose = st.number_input('Glucose' , min_value=0.0 , max_value=10.0,value=1.0)
Insulin =  st.number_input('Insulin' , min_value=0.0 , max_value=100.0,value=1.0)
BMI =  st.number_input('BMI' , min_value=0.0 , max_value=100.0,value=1.0)
DiabetesPedigreeFunction =  st.number_input('DiabetesPedigreeFunction' , min_value=0.0 , max_value=100.0,value=1.0)
Age =  st.number_input('Age' , min_value=0.0 , max_value=100.0,value=1.0)

output = model.predict([[Pregnancies,Glucose,Insulin,BMI,DiabetesPedigreeFunction,Age]])

#display the result
st.write("diabetes patient : ",round(output[0],2))
