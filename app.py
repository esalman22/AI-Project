import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and assets
@st.cache_resource
def load_assets():
    model = joblib.load('anemia_rf_model.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    features = joblib.load('features.pkl')
    return model, encoders, target_encoder, features

model, encoders, target_encoder, features = load_assets()

st.set_page_config(page_title="Anemia Prediction System", page_icon="🩸")

st.title("🩸 Anemia Prediction System")
st.markdown("""
This application predicts the type of anemia based on clinical parameters using a Random Forest model.
""")

st.sidebar.header("Patient Data Input")

def user_input_features():
    # Numerical inputs
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.0)
    rdw = st.sidebar.number_input("RDW (%)", 10.0, 25.0, 14.0)
    mcv = st.sidebar.number_input("MCV (fL)", 50.0, 120.0, 85.0)
    age = st.sidebar.number_input("Age", 0, 120, 30)
    
    # Categorical inputs
    gender = st.sidebar.selectbox("Gender", encoders['Gender'].classes_)
    
    # Filter pregnancy status based on gender
    if gender == 'Male':
        pregnancy_status = 'Not Applicable'
    else:
        pregnancy_status = st.sidebar.selectbox("Pregnancy Status", encoders['Pregnancy_Status'].classes_)
        
    medical_history = st.sidebar.selectbox("Medical History", encoders['Medical_History'].classes_)
    activity_level = st.sidebar.selectbox("Activity Level", encoders['Activity_Level'].classes_)
    tea_coffee_intake = st.sidebar.selectbox("Tea/Coffee Intake", encoders['Tea_Coffee_Intake'].classes_)
    
    # Derived features (matching training logic)
    hb_level = 'Low' if hemoglobin < 11 else ('Normal' if hemoglobin < 13 else 'High')
    cell_status = 'Microcytic' if mcv < 80 else ('Normocytic' if mcv < 100 else 'Macrocytic')
    blood_score = (hemoglobin + rdw + mcv) / 3
    
    data = {
        'Hemoglobin': hemoglobin,
        'RDW': rdw,
        'MCV': mcv,
        'Age': age,
        'Gender': gender,
        'Pregnancy_Status': pregnancy_status,
        'Medical_History': medical_history,
        'Activity_Level': activity_level,
        'Tea_Coffee_Intake': tea_coffee_intake,
        'Hb_Level': hb_level,
        'Cell_Status': cell_status,
        'Blood_Score': blood_score
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Patient Parameters")
st.write(input_df)

# Preprocessing input
processed_df = input_df.copy()
for col in ['Gender', 'Pregnancy_Status', 'Medical_History', 'Activity_Level', 'Tea_Coffee_Intake', 'Hb_Level', 'Cell_Status']:
    processed_df[col] = encoders[col].transform(input_df[col])

# Ensure feature order matches training
processed_df = processed_df[features]

# Prediction
if st.button("Predict Anemia Type"):
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)
    
    result = target_encoder.inverse_transform(prediction)[0]
    
    result_map = {
        'ID': 'Iron Deficiency (ID)',
        'IDA': 'Iron Deficiency Anemia (IDA)',
        'Normal': 'Normal (No Anemia)'
    }
    display_result = result_map.get(result, result)
    
    st.subheader("Prediction Result")
    if result == 'Normal':
        st.success(f"The predicted condition is: **{display_result}**")
    else:
        st.warning(f"The predicted condition is: **{display_result}**")
    
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=target_encoder.classes_)
    prob_df.columns = [result_map.get(c, c) for c in prob_df.columns]
    st.bar_chart(prob_df.T)

st.info("This model uses all clinical features including derived scores for maximum accuracy.")
