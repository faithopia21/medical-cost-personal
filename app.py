import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Medical Cost Predictor", layout="centered")

st.title("Medical Insurance Cost Predictor")
st.write("Estimate medical insurance charges based on patient information.")

model = joblib.load('models/medical_cost_model.joblib')
scaler = joblib.load('models/scaler.joblib')

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30)
sex = st.sidebar.selectbox("Sex", ["female", "male"])
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0],
    })

    numeric_cols = ['age', 'bmi', 'children']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    input_df['smoker_bmi'] = input_df['smoker_yes'] * input_df['bmi']

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Annual Medical Cost")
    st.metric(label="Estimated Charges", value=f"${prediction:,.2f}")