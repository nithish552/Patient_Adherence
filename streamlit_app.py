import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    # high_level_image = Image.open("_assets/high_level_overview.png")
    # st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    # st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
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
    if(st.button('Predict')):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                pred
            )
        )
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()
