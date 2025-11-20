#Created by Dhiraj using streamlit. This is the UI of Heart Disease Prediction MOdel:

#setup
import streamlit as st
import pandas as pd
import joblib

def load_assets():
    """Loads and returns the model, scaler, and expected columns."""
    try:
        model = joblib.load("KNN_heart_model.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("Error: One or more asset files (.pkl) not found. Please ensure the model, scaler, and columns files are in the correct directory.")
        return None, None, None

# Load assets and exit if they are not found
model, scaler, expected_columns = load_assets()
if not all([model, scaler, expected_columns]):
    st.stop()

#creating title 
st.title("Heart Stroke Prediction by Dhiraj 👌")
st.markdown("Provide the following details to check your heart stroke risk:")

#collecting user Input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

def predict_risk(model, scaler, data):
    """Scales the input data and makes a prediction."""
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return prediction[0]

# When Predict is clicked
if st.button("Predict"):

    # Create a dictionary from user inputs
    user_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak
    }

    # Create a DataFrame with all expected columns initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Populate the DataFrame with the user's input
    for key, value in user_input.items():
        if key in input_df.columns:
            input_df.at[0, key] = value

    # Set the one-hot encoded columns based on user selection
    input_df.at[0, 'Sex_' + sex] = 1
    input_df.at[0, 'ChestPainType_' + chest_pain] = 1
    input_df.at[0, 'RestingECG_' + resting_ecg] = 1
    input_df.at[0, 'ExerciseAngina_' + exercise_angina] = 1
    input_df.at[0, 'ST_Slope_' + st_slope] = 1

    prediction = predict_risk(model, scaler, input_df)

    # Show result
    if prediction == 1:
        st.warning("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")