import streamlit as st
import joblib
import pandas as pd

# Load the trained Naive Bayes model
model = joblib.load('naive_bayes_model.pkl')

# Streamlit app title
st.title("Feeding Intolerance Prediction")

# User input form for feature values
age = st.number_input("Age", min_value=0, max_value=100, value=30)
comorbid = st.selectbox("Comorbid (Yes/No)", ["Yes", "No"])
asa = st.selectbox("ASA score", [1, 2, 3, 4])
duration = st.number_input("Duration of Procedure (minutes)", min_value=0, max_value=500, value=100)
laparoscopic = st.selectbox("Laparoscopic Procedure (Open/Laparoscopic)", ["Open", "Laparoscopic"])
bloodloss = st.number_input("Bloodloss (ml)", min_value=0, max_value=1000, value=100)
epidural = st.selectbox("Epidural (Yes/No)", ["Yes", "No"])

# Preprocess the inputs
comorbid = 1 if comorbid == "Yes" else 0
laparoscopic = 1 if laparoscopic == "Laparoscopic" else 0
epidural = 1 if epidural == "Yes" else 0

# Create the input data as a dataframe
input_data = pd.DataFrame([[age, comorbid, asa, duration, laparoscopic, bloodloss, epidural]],
                          columns=['age', 'comorbid', 'asa', 'duration', 'laparoscopic', 'bloodloss', 'epidural'])

# Make the prediction
prediction = model.predict(input_data)

# Display prediction result
if prediction == 1:
    st.write("Prediction: **Intolerance** (Feeding Intolerance is likely)")
    st.write("Recommendation: You may want to consider further medical evaluation and monitoring.")
else:
    st.write("Prediction: **No Intolerance** (Feeding Intolerance is unlikely)")
    st.write("Recommendation: The patient seems stable for enteral feeding. Continue routine monitoring.")
