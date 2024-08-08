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
from PIL import Image

st.set_page_config(page_title="Predicting Adherence", page_icon="📈", layout='wide')

# Add custom CSS for background and animations
st.markdown(
    """<style>
        body {
            background: linear-gradient(to bottom right, #4e54c8, #8f94fb);
            color: #fff;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background: #6a67ce;
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s, transform 0.3s;
        }
        .stButton>button:hover {
            background-color: #5756d6;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            border: 2px solid #6a67ce;
            border-radius: 5px;
            padding: 10px;
            color: #333;
        }
        .stSelectbox>div>div>div>div>div>div>div>div>div {
            border: 2px solid #6a67ce;
            border-radius: 5px;
            padding: 10px;
            color: #333;
        }
        .stNumberInput>div>div>div>div>div>input {
            border: 2px solid #6a67ce;
            border-radius: 5px;
            padding: 10px;
            color: #333;
        }
        .stApp {
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>"""
    , unsafe_allow_html=True)

if st.session_state.get('logged_in'):
    st.title("Embedded Power BI Dashboard")
    powerbi_report_url = "https://app.powerbi.com/reportEmbed?reportId=a97f5efa-7c3a-4430-bc10-b70154171e32&autoAuth=true&ctid=ac29cc87-d08e-4f42-bee7-3223d3c25249"
    st.components.v1.iframe(powerbi_report_url, width=1000, height=600)

    predicted_adherence = ""

    # Load the preprocessor and model
    with open('preprocessor.pkl', 'rb') as preproc_file, open('model.pkl', 'rb') as model_file:
        preprocessor = pickle.load(preproc_file)
        model = pickle.load(model_file)

    # Load the label encoder for Adherence
    with open('label_encoder.pkl', 'rb') as le_file:
        le_y = pickle.load(le_file)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    st.title('Patient Adherence Prediction')
    st.write("Provide the following patient details:")

    # Organize input fields into columns
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col7, col8 = st.columns(2)
    col9, col10 = st.columns(2)
    col11, col12 = st.columns(2)
    col13, col14 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
    with col2:
        gender = st.selectbox("Gender", ("Male", "Female"))
    with col3:
        race = st.selectbox("Race", ("ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN", "OTHER"))
    with col4:
        insurance_type = st.selectbox("Insurance Type", ("COMMERCIAL", "NON-COMMERCIAL"))
    with col5:
        median_income = st.number_input('Median Income', min_value=0, value=50000)
    with col6:
        hospitalization_prior_year = st.selectbox("Hospitalization Prior Year", ("YES", "NO"))
    with col7:
        ms_related_hospitalization = st.selectbox("MS Related Hospitalization", ("YES", "NO"))
    with col8:
        relapse_prior_year = st.selectbox("Relapse Prior Year", ("YES", "NO"))
    with col9:
        disease = st.selectbox("Disease", ("BIPOLAR 1 DISORDER", "ASTHMA", "HYPERTENSION", "DIABETES MELLITUS"))
    with col10:
        therapeutic_area = st.selectbox("Therapeutic Area", ("PSYCHIATRY", "PULMONOLOGY", "CARDIOLOGY", "ENDOCRINOLOGY"))
    with col11:
        specialty_pharma = st.selectbox("Specialty Pharma", ("LITHIUM", "INHALED CORTICOSTEROIDS", "ACE INHIBITORS", "INSULIN"))
    with col12:
        trial_length_weeks = st.number_input('Trial Length (Weeks)', min_value=0, value=12)
    with col13:
        micro_reimbursements = st.selectbox('Micro Reimbursements', ['Yes', 'No'])
    with col14:
        dose_length_seconds = st.number_input('Dose Length (Seconds)', min_value=0, value=60)
    
    dose_delay_hours = st.number_input('Dose Delay (Hours)', min_value=0, value=2)

    medication_name = st.selectbox("Medication Name", ("UNKNOWN", "DULERA", "BENICAR HCT", "BENICAR", "ALTACE",
                                                       "ADVAIR", "PULMICORT", "PRINZIDE", "ALBUTEROL", "NORVASC",
                                                       "SINGULAIR", "QVAR", "ALVESCO", "COREG", "AVAPRO", "SYMBICORT",
                                                       "ASMANEX", "ZESTORETIC", "FLOVENT", "MICARDIS HCT", "ZESTRIL",
                                                       "MICARDIS", "COZAAR", "DIOVAN", "UNIVASC", "HYZAAR", "LOTREL",
                                                       "PRINIVIL", "AVALIDE"))

    brand_name = st.selectbox("Brand Name", ("UNKNOWN", "MOMETASONE FUROATE AND FORMOTEROL FUMARATE",
                                             "OLMESARTAN MEDOXOMIL AND HYDROCHLOROTHIAZIDE", "OLMESARTAN MEDOXOMIL",
                                             "RAMIPRIL", "FLUTICASONE PROPIONATE AND SALMETEROL", "BUDESONIDE",
                                             "LISINOPRIL/HYDROCHLOROTHIAZIDE", "ALBUTEROL SULFATE", "AMLODIPINE BESYLATE",
                                             "MONTELUKAST SODIUM", "BECLOMETHASONE DIPROPIONATE", "CICLESONIDE",
                                             "CARVEDILOL", "IRBESARTAN", "BUDESONIDE AND FORMOTEROL FUMARATE DIHYDRATE",
                                             "MOMETASONE FUROATE", "LISINOPRIL AND HYDROCHLOROTHIAZIDE",
                                             "FLUTICASONE PROPIONATE", "TELMISARTAN AND HYDROCHLOROTHIAZIDE",
                                             "LISINOPRIL", "TELMISARTAN", "LOSARTAN POTASSIUM", "VALSARTAN",
                                             "MOEXIPRIL HCL", "LOSARTAN POTASSIUM/HYDROCHLOROTHIAZIDE", 
                                             "AMLODIPINE BESYLATE AND BENAZEPRIL HCL", "IRBESARTAN AND HYDROCHLOROTHIAZIDE"))

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

    if st.button('Predict'):
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)
        predicted_adherence = le_y.inverse_transform(prediction)[0]
        st.write(f'Predicted Adherence: {predicted_adherence}')

        if predicted_adherence == "NON-ADHERENT":
            client = Client("yuva2110/vanilla-charbot")
            result = client.predict(
                message=f"give recommendations for patients with {disease} in therapeutic area of {therapeutic_area} taking speciality pharma of {specialty_pharma} and medication {medication_name} to continue there medication to avoid future expenses and say them about the risks involved with discontinuation in detailed manner and make them retained and give name as {brand_name} , give in points focus mainly on  retaining them and give appropriate advice and suggestions",
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
