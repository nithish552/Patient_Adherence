import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('saved_model\model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input fields for the form
st.title('Patient Adherence Prediction')
st.write("Provide the following patient details:")

age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
race = st.text_input('Race')
insurance_type = st.selectbox('Insurance Type', ['Commercial', 'Non-commercial'])
median_income = st.number_input('Median Income', min_value=0, value=50000)
hospitalization_prior_year = st.selectbox('Hospitalization Prior Year', ['Yes', 'No'])
ms_related_hospitalization = st.selectbox('MS Related Hospitalization', ['Yes', 'No'])
relapse_prior_year = st.selectbox('Relapse Prior Year', ['Yes', 'No'])
disease = st.text_input('Disease')
therapeutic_area = st.text_input('Therapeutic Area')
specialty_pharma = st.text_input('Specialty Pharma')
adherence = st.selectbox('Adherence', ['High', 'Medium', 'Low'])
trial_length_weeks = st.number_input('Trial Length (Weeks)', min_value=0, value=12)
micro_reimbursements = st.number_input('Micro Reimbursements', min_value=0, value=0)
dose_length_seconds = st.number_input('Dose Length (Seconds)', min_value=0, value=0)
dose_delay_hours = st.number_input('Dose Delay (Hours)', min_value=0, value=0)
medication_name = st.text_input('Medication Name')
brand_name = st.text_input('Brand Name')

# Collect the input data
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Race': [race],
    'InsuranceType': [insurance_type],
    'MedianIncome': [median_income],
    'HospitalizationPriorYear': [hospitalization_prior_year],
    'MSRelatedHospitalization': [ms_related_hospitalization],
    'RelapsePriorYear': [relapse_prior_year],
    'Disease': [disease],
    'TherapeuticArea': [therapeutic_area],
    'SpecialtyPharma': [specialty_pharma],
    'Adherence': [adherence],
    'TrialLengthWeeks': [trial_length_weeks],
    'MicroReimbursements': [micro_reimbursements],
    'DoseLengthSeconds': [dose_length_seconds],
    'DoseDelayHours': [dose_delay_hours],
    'MedicationName': [medication_name],
    'BrandName': [brand_name]
})

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted outcome: {prediction[0]}')

