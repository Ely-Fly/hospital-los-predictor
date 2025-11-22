import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'length_of_stay_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload the model file to your repository.")
        st.stop()
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# Feature columns (must match training order)
FEATURE_COLUMNS = [
    'vdate', 'rcount', 'gender', 'dialysisrenalendstage', 'asthma',
    'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor',
    'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
    'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
    'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9',
    'facid_B', 'facid_C', 'facid_D', 'facid_E'
]

def predict_los(input_data: dict) -> float:
    """Predict Length of Stay from patient features"""
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_COLUMNS]
    prediction = model.predict(input_df)
    return float(prediction[0])

# Streamlit App
st.title("🏥 Hospital Length of Stay Predictor")
st.write("Enter patient information to predict the expected length of stay in days.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Patient Info", "🧪 Lab Results", "🏥 Facility"])

with tab1:
    st.subheader("Demographics & Visit Information")
    col1, col2 = st.columns(2)
    
    with col1:
        vdate = st.selectbox("Visit Day of Week", 
                            options=[0, 1, 2, 3, 4, 5, 6],
                            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        rcount = st.number_input("Readmission Count", min_value=0, max_value=5, value=0, step=1)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        pulse = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=200, value=80, step=1)
        respiration = st.number_input("Respiration Rate", min_value=5, max_value=50, value=18, step=1)
    
    st.subheader("Medical Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dialysisrenalendstage = st.checkbox("Dialysis/Renal End Stage")
        asthma = st.checkbox("Asthma")
        irondef = st.checkbox("Iron Deficiency")
        pneum = st.checkbox("Pneumonia")
    
    with col2:
        substancedependence = st.checkbox("Substance Dependence")
        psychologicaldisordermajor = st.checkbox("Major Psychological Disorder")
        depress = st.checkbox("Depression")
        psychother = st.checkbox("Psychotherapy")
    
    with col3:
        fibrosisandother = st.checkbox("Fibrosis & Other")
        malnutrition = st.checkbox("Malnutrition")
        secondarydiagnosisnonicd9 = st.number_input("Secondary Diagnoses (non-ICD9)", min_value=0, max_value=20, value=0, step=1)

with tab2:
    st.subheader("Laboratory Results")
    col1, col2 = st.columns(2)
    
    with col1:
        hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=13.5, step=0.1)
        hematocrit = st.number_input("Hematocrit (%)", min_value=20.0, max_value=60.0, value=40.0, step=0.1)
        neutrophils = st.number_input("Neutrophils", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        sodium = st.number_input("Sodium (mEq/L)", min_value=120.0, max_value=160.0, value=140.0, step=0.1)
    
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=400.0, value=100.0, step=1.0)
        bloodureanitro = st.number_input("Blood Urea Nitrogen (mg/dL)", min_value=5.0, max_value=100.0, value=15.0, step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=10.0, value=1.0, step=0.1)

with tab3:
    st.subheader("Facility Information")
    st.write("Select the facility (Facility A is the reference category)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        facid_B = st.checkbox("Facility B")
    with col2:
        facid_C = st.checkbox("Facility C")
    with col3:
        facid_D = st.checkbox("Facility D")
    with col4:
        facid_E = st.checkbox("Facility E")

# Prediction button
st.divider()
if st.button("🔮 Predict Length of Stay", type="primary", use_container_width=True):
    # Collect all input data
    input_data = {
        'vdate': vdate,
        'rcount': rcount,
        'gender': gender,
        'dialysisrenalendstage': int(dialysisrenalendstage),
        'asthma': int(asthma),
        'irondef': int(irondef),
        'pneum': int(pneum),
        'substancedependence': int(substancedependence),
        'psychologicaldisordermajor': int(psychologicaldisordermajor),
        'depress': int(depress),
        'psychother': int(psychother),
        'fibrosisandother': int(fibrosisandother),
        'malnutrition': int(malnutrition),
        'hemo': hemo,
        'hematocrit': hematocrit,
        'neutrophils': neutrophils,
        'sodium': sodium,
        'glucose': glucose,
        'bloodureanitro': bloodureanitro,
        'creatinine': creatinine,
        'bmi': bmi,
        'pulse': pulse,
        'respiration': respiration,
        'secondarydiagnosisnonicd9': secondarydiagnosisnonicd9,
        'facid_B': int(facid_B),
        'facid_C': int(facid_C),
        'facid_D': int(facid_D),
        'facid_E': int(facid_E)
    }
    
    # Make prediction
    predicted_los = predict_los(input_data)
    
    # Display result
    st.success(f"### Predicted Length of Stay: **{predicted_los:.2f} days**")
    
    # Additional context
    if predicted_los < 3:
        st.info("📊 This is a relatively short stay.")
    elif predicted_los < 7:
        st.info("📊 This is a moderate length stay.")
    else:
        st.warning("📊 This is a longer than average stay.")

# Footer
st.divider()
st.caption("⚠️ This prediction is for informational purposes only and should not replace clinical judgment.")
