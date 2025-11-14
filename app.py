import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.title('Hypertension Prediction App')
st.write('This app predicts whether a person has hypertension based on health and lifestyle factors.')

model = joblib.load(r'C:\Users\USER\PycharmProjects\ML Reinforcement\Model\trained_model.pkl')
scaler = joblib.load(r'C:\Users\USER\PycharmProjects\ML Reinforcement\Model\fixed_scaler.pkl')

st.header('Enter the details below:')

Age = st.number_input('Age', min_value=18, max_value=None, value=30)
Salt_intake = st.number_input('Salt Intake (G/Day)', min_value=0.0, max_value=None, value=20.0)
Stress_score = st.slider("Stress Score", 0, 10, 5)
BP_history = st.selectbox("BP History", ['Normal', 'Prehypertension', 'Hypertension'])
Sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=10.0)
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
Medication = st.selectbox("Medication", ['No Medication', 'ACE Inhibitor', 'Beta Blocker', 'Other'])
Family_history = st.selectbox("Family History", ['Yes', 'No'])
Exercise_level = st.selectbox("Exercise Level", ['Low', 'Moderate', 'High'])
Smoking_status = st.selectbox("Smoking Status", ['Smoker', 'Non-Smoker'])


bp_map = {'Normal': 0, 'Prehypertension': 1, 'Hypertension': 2}
med_map = {'No Medication': 0, 'ACE Inhibitor': 1, 'Beta Blocker': 2, 'Other': 3}
fam_map = {'No': 0, 'Yes': 1}
ex_map = {'Low': 0, 'Moderate': 1, 'High': 2}
smoke_map = {'Non-Smoker': 0, 'Smoker': 1}

input_data = np.array([[Age, Salt_intake, Stress_score, bp_map[BP_history],
                        Sleep_duration, BMI, med_map[Medication],
                        fam_map[Family_history], ex_map[Exercise_level],
                        smoke_map[Smoking_status]]])

input_scaled =scaler.transform(input_data)
if st.button("Predict Hypertension"):
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.write(f'model confidence(Hypertension): {prob:.2f}')

    if prediction == 1:
        st.error("The model predicts that this person **has hypertension.**")
    else:
        st.success("The model predicts that this person **does not have hypertension.**")