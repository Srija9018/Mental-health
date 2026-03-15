import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("model.pkl", "rb"))

st.title("Workplace Mental Health Support-Seeking Prediction")

st.write("Predict whether an employee will seek mental health support.")

# Inputs
Age = st.number_input("Age", 18, 65)

Gender = st.selectbox("Gender", ["Male", "Female", "Other"])

Country = st.selectbox("Country", ["United States", "India", "Canada", "UK"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

family_history = st.selectbox("Family History", ["Yes", "No"])

treatment = st.selectbox("Treatment", ["Yes", "No"])

work_interfere = st.selectbox(
    "Work Interfere",
    ["Never", "Rarely", "Sometimes", "Often"]
)

remote_work = st.selectbox("Remote Work", ["Yes", "No"])

tech_company = st.selectbox("Tech Company", ["Yes", "No"])
seek_help = st.selectbox("Company Encourages Seeking Help",["Yes","No"])


# Encoding
Gender = 1 if Gender=="Male" else 0
Country=1 if Country=="United States" else 0
self_employed=1 if self_employed=="Yes" else 0
family_history = 1 if family_history=="Yes" else 0
treatment=1 if treatment=="Yes" else 0
work_interfere = ["Never","Rarely","Sometimes","Often"].index(work_interfere)
remote_work = 1 if remote_work=="Yes" else 0
tech_company=1 if tech_company=="Yes" else 0
seek_help = 1 if seek_help=="Yes" else 0

features = np.array([[Age,Gender,Country,self_employed,family_history,treatment,work_interfere,remote_work,tech_company,seek_help]])

# Prediction
if st.button("Predict"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Employee is likely to seek mental health support")
    else:
        st.warning("Employee is unlikely to seek mental health support")