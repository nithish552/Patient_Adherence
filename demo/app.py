import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from gradio_client import Client

predicted_adherence = ""

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as preproc_file, open('model.pkl', 'rb') as model_file:
    preprocessor = pickle.load(preproc_file)
    model = pickle.load(model_file)

# Load the label encoder for Adherence
with open('label_encoder.pkl', 'rb') as le_file:
    le_y = pickle.load(le_file)

# Create a pipeline with the preprocessor and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Define the input fields for the form
st.title('Patient Adherence Prediction')
st.write("Provide the following patient details:")

age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
race = st.text_input('Race',value = 'Alien')
insurance_type = st.text_input('Insurance Type', value='Commercial')
median_income = st.number_input('Median Income', min_value=0, value=50000)
hospitalization_prior_year = st.selectbox('Hospitalization Prior Year', ['Yes', 'No', np.nan])
ms_related_hospitalization = st.selectbox('MS Related Hospitalization', ['Yes', 'No'])
relapse_prior_year = st.selectbox('Relapse Prior Year', ['Yes', 'No'])
disease = st.text_input('Disease', value = 'Diabetes Mellitus')
therapeutic_area = st.text_input('Therapeutic Area', value = 'Endocrinology')
specialty_pharma = st.text_input('Specialty Pharma', value = 'Insulin')
trial_length_weeks = st.number_input('Trial Length (Weeks)', min_value=0, value=12)
micro_reimbursements = st.selectbox('Micro Reimbursements', ['Yes', 'No'])
dose_length_seconds = st.number_input('Dose Length (Seconds)', min_value=0, value=60)
dose_delay_hours = st.number_input('Dose Delay (Hours)', min_value=0, value=2)
medication_name = st.text_input('Medication Name',value = 'PRINIVIL')
brand_name = st.text_input('Brand Name', value = 'LISINOPRIL')

# Collect the input data into a DataFrame
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
    'TrialLengthWeeks': [trial_length_weeks],
    'MicroReimbursements': [micro_reimbursements],
    'DoseLengthSeconds': [dose_length_seconds],
    'DoseDelayHours': [dose_delay_hours],
    'MedicationName': [medication_name],
    'BrandName': [brand_name]
})

# Preprocess the input data and predict
if st.button('Predict'):
    # Transform the data using the preprocessor
    transformed_data = pipeline['preprocessor'].transform(input_data)
    # Predict using the model
    prediction = pipeline['classifier'].predict(transformed_data)
    # Decode the prediction
    predicted_adherence = le_y.inverse_transform(prediction)[0]
    # Display the prediction
    st.write(f'Predicted Adherence: {predicted_adherence}')

if(predicted_adherence == "ADHERENT"):
    client = Client("yuva2110/vanilla-charbot")
    result = client.predict(
            message=f"give recommendations for patients with {disease} in therapeutic area of {therapeutic_area} taking speciality pharma of {specialty_pharma} and medication {medication_name} to continue there medication to avoid future expenses and say them about the risks involved with discontinuation in detailed manner and make them retained and give name as {brand_name} , give in points",
            # message = "Hello!!",
            system_message="You are a friendly Chatbot.",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            api_name="/chat"
    )
    st.write(result)
