import streamlit as st
import pandas as pd
import numpy as np
import joblib

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Medical Cost Predictor", layout="centered")

st.title("Medical Insurance Cost Predictor")
st.write("Estimate medical insurance charges based on patient information.")

model = joblib.load('models/medical_cost_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@st.cache_data
def load_background_data():
    df = pd.read_csv('data/insurance.csv')
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = ['age', 'bmi', 'children']
    bg_scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = bg_scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = bg_scaler.transform(X_test[numeric_cols])

    X_train_scaled['smoker_bmi'] = X_train_scaled['smoker_yes'] * X_train_scaled['bmi']
    X_test_scaled['smoker_bmi'] = X_test_scaled['smoker_yes'] * X_test_scaled['bmi']

    return X_train_scaled, X_test_scaled, y_test

X_train_scaled, X_test_scaled, y_test = load_background_data()

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
st.sidebar.caption("BMI = weight (kg) ÷ height (m)². Example: 70kg ÷ (1.7m)² ≈ 24.2")
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'Male' else 0],
        'smoker_yes': [1 if smoker == 'Yes' else 0],
        'region_northwest': [1 if region == 'Northwest' else 0],
        'region_southeast': [1 if region == 'Southeast' else 0],
        'region_southwest': [1 if region == 'Southwest' else 0],
    })

    numeric_cols = ['age', 'bmi', 'children']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    input_df['smoker_bmi'] = input_df['smoker_yes'] * input_df['bmi']

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Annual Medical Cost")
    st.metric(label="Estimated Charges", value=f"${prediction:,.2f}")

    # SHAP explanation
    st.subheader("What drove this prediction")

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

st.divider()
st.subheader("Model Information")

y_pred_test = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Algorithm**")
    st.write("Multiple Linear Regression with smoker × BMI interaction term")

    st.markdown("**Dataset**")
    st.write("Medical Cost Personal Datasets (Kaggle)")
    st.markdown("[View dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)")

with col2:
    st.markdown("**Test Set Performance**")
    st.metric("R²", f"{r2:.4f}")
    st.caption("Proportion of cost variation the model explains. Closer to 1.0 is better.")

    st.metric("MAE", f"${mae:,.2f}")
    st.caption("Average prediction error in dollars, regardless of direction.")

    st.metric("RMSE", f"${rmse:,.2f}")
    st.caption("Average prediction error in dollars, weighted more heavily toward large misses.")