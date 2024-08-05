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

# Load the preprocessor, model, and label encoder
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define the Streamlit app
def main():
    st.title("Patient Adherence Prediction")
    st.write("Enter patient information to predict adherence status.")

    # Input fields for patient information
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", options=["FEMALE", "MALE"])
    race = st.selectbox("Race", options=["ASIAN", "BLACK", "WHITE", "OTHER"])
    insurance_type = st.selectbox("Insurance Type", options=["COMMERCIAL", "NON-COMMERCIAL"])
    median_income = st.number_input("Median Income", min_value=0)
    hospitalization_prior_year = st.selectbox("Hospitalization Prior Year", options=["YES", "NO"])
    ms_related_hospitalization = st.selectbox("MS Related Hospitalization", options=["YES", "NO"])
    relapse_prior_year = st.selectbox("Relapse Prior Year", options=["YES", "NO"])
    disease = st.text_input("Disease")
    therapeutic_area = st.text_input("Therapeutic Area")
    specialty_pharma = st.text_input("Specialty Pharma")
    trial_length_weeks = st.number_input("Trial Length (Weeks)", min_value=0)
    micro_reimbursements = st.selectbox("Micro Reimbursements", options=["YES", "NO"])
    dose_length_seconds = st.number_input("Dose Length (Seconds)", min_value=0.0)
    dose_delay_hours = st.number_input("Dose Delay (Hours)", min_value=0.0)
    medication_name = st.text_input("Medication Name")
    brand_name = st.text_input("Brand Name")

    # Prediction button
    if st.button("Predict"):
        # Create a DataFrame for the custom input
        custom_input = {
            'Age': age,
            'Gender': gender,
            'Race': race,
            'InsuranceType': insurance_type,
            'MedianIncome': median_income,
            'HospitalizationPriorYear': hospitalization_prior_year,
            'MSRelatedHospitalization': ms_related_hospitalization,
            'RelapsePriorYear': relapse_prior_year,
            'Disease': disease,
            'TherapeuticArea': therapeutic_area,
            'SpecialtyPharma': specialty_pharma,
            'TrialLengthWeeks': trial_length_weeks,
            'MicroReimbursements': micro_reimbursements,
            'DoseLengthSeconds': dose_length_seconds,
            'DoseDelayHours': dose_delay_hours,
            'MedicationName': medication_name,
            'BrandName': brand_name
        }

        custom_input_df = pd.DataFrame([custom_input])

        # Ensure preprocessor is correctly loaded as a ColumnTransformer
        if isinstance(preprocessor, Pipeline):
            # Preprocess the custom input
            processed_input = preprocessor.transform(custom_input_df)

            # Make prediction using the trained model
            prediction = model.predict(processed_input)

            # Decode the prediction back to original labels
            prediction_label = label_encoder.inverse_transform(prediction)

            # Display the prediction result
            st.write(f"Prediction: {prediction_label[0]}")
        else:
            st.error("Error: The preprocessor object is not correctly loaded.")

if __name__ == '__main__':
    main()
