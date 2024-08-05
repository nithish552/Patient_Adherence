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
from session import get_session
import time
st.set_page_config(page_title="Predicting Adherence", page_icon="📈",layout='wide')
print(st.session_state)
if st.session_state.get('logged_in'):
    print(st.session_state)
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
    gender = st.selectbox("gender",("Male","Female"))
    race = st.selectbox("Race",("ASIAN","WHITE","BLACK OR AFRICAN AMERICAN","OTHER"))
    insurance_type = st.selectbox("InsuranceType",("COMMERCIAL","NON-COMMERCIAL"))
    median_income = st.number_input('Median Income', min_value=0, value=50000)
    hospitalization_prior_year = st.selectbox("HospitalizationPriorYear",("YES","NO"))
    ms_related_hospitalization = st.selectbox("MSRelatedHospitalization",("YES","NO"))
    relapse_prior_year = st.selectbox("RelapsePriorYear",("YES","NO"))
    disease = st.selectbox("Disease",("BIPOLAR 1 DISORDER","ASTHMA","HYPERTENSION","DIABETES MELLITUS"))
    therapeutic_area = st.selectbox("TherapeuticArea",("PSYCHIATRY","PULMONOLOGY","CARDIOLOGY","ENDOCRINOLOGY"))
    specialty_pharma = st.selectbox("SpecialtyPharma",("LITHIUM","INHALED CORTICOSTEROIDS","ACE INHIBITORS","ACE INHIBITORS", "INSULIN"))
    trial_length_weeks = st.number_input('Trial Length (Weeks)', min_value=0, value=12)
    micro_reimbursements = st.selectbox('Micro Reimbursements', ['Yes', 'No'])
    dose_length_seconds = st.number_input('Dose Length (Seconds)', min_value=0, value=60)
    dose_delay_hours = st.number_input('Dose Delay (Hours)', min_value=0, value=2)
    medication_name = st.selectbox("MedicationName",("UNKNOWN","DULERA","BENICAR HCT","BENICAR","ALTACE","ADVAIR","PULMICORT","PRINZIDE","ALBUTEROL","NORVASC","SINGULAIR","QVAR","ALVESCO","COREG","AVAPRO","SYMBICORT","ASMANEX","ZESTORETIC","FLOVENT","MICARDIS HCT","ZESTRIL","MICARDIS","COZAAR","DIOVAN","UNIVASC","HYZAAR","LOTREL","PRINIVIL","AVALIDE"))
    brand_name = st.selectbox("BrandName",("UNKNOWN","MOMETASONE FUROATE AND FORMOTEROL FUMARATE","OLMESARTAN MEDOXOMIL AND HYDROCHLOROTHIAZIDE","OLMESARTAN MEDOXOMIL","RAMIPRIL","FLUTICASONE PROPIONATE AND SALMETEROL","BUDESONIDE","LISINOPRIL/HYDROCHLOROTHIAZIDE","ALBUTEROL SULFATE","AMLODIPINE BESYLATE","MONTELUKAST SODIUM","BECLOMETHASONE DIPROPIONATE","CICLESONIDE","CARVEDILOL","IRBESARTAN","BUDESONIDE AND FORMOTEROL FUMARATE DIHYDRATE","MOMETASONE FUROATE","LISINOPRIL AND HYDROCHLOROTHIAZIDE","FLUTICASONE PROPIONATE","TELMISARTAN AND HYDROCHLOROTHIAZIDE","LISINOPRIL","TELMISARTAN","LOSARTAN POTASSIUM","VALSARTAN","MOEXIPRIL HCL","LOSARTAN POTASSIUM/HYDROCHLOROTHIAZIDE","AMLODIPINE BESYLATE AND BENAZEPRIL HCL","IRBESARTAN AND HYDROCHLOROTHIAZIDE"))

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
        # custom_input_df = pd.DataFrame([custom_input])

        # Preprocess the custom input
        processed_input = preprocessor.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(processed_input)

        # Decode the prediction back to original labels
        # prediction_label = label_encoder.inverse_transform(prediction)

        # # Display the prediction result
        # st.write(f"Prediction: {prediction_label[0]}")
        
        # # Transform the data using the preprocessor
        # transformed_data = pipeline['preprocessor'].transform(input_data)
        # # Predict using the model
        # prediction = pipeline['classifier'].predict(transformed_data)
        # # Decode the prediction
        predicted_adherence = le_y.inverse_transform(prediction)[0]
        # Display the prediction
        st.write(f'Predicted Adherence: {predicted_adherence}')

    if(predicted_adherence == "NON-ADHERENT"):
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
    else:
        client = Client("yuva2110/vanilla-charbot")
        result = client.predict(
                message=f"give appreciation for patients with {disease} in therapeutic area of {therapeutic_area} taking speciality pharma of {specialty_pharma} and medication {medication_name} to continue there medication to avoid future expenses and say them about the risks involved with discontinuation in detailed manner and make them retained and feel good about their progress and give name as {brand_name} , give in points",
                # message = "Hello!!",
                system_message="You are a friendly Chatbot.",
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                api_name="/chat"
        )
    st.write(result)
else:
    st.error("You should sign in")
    time.sleep(1)
    st.switch_page('pages/login.py')