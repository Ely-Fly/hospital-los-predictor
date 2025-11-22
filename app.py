#### goal: analyze hospital length of stay dataset
# predict the length of stay 

import pandas as pd
import numpy as np

# data description:  https://microsoft.github.io/r-server-hospital-length-of-stay/input_data.html 
data = pd.read_csv('/Users/ellayuan/Desktop/Cursor/lesson3/LengthOfStay.csv')



print(data.info()) # no null values 

print(data['lengthofstay'].describe())

# # categorical feature exploration 
print(data['gender'].value_counts())



# data transformation 
#a. five objects: eid, vdate, rcount, gener, discharged, facid
data['vdate'] = pd.to_datetime(data['vdate'])
print(data['vdate'])
data['vdate'] = data['vdate'].dt.dayofweek
print(data['vdate'])

data.drop(columns = ['discharged', 'eid'], inplace = True)

# print(data.head())
# print(data.info())

#b. handing complex category data
# convert 5+ to 5 and then convert the entire column to a numeric type 
data['rcount'] = data['rcount'].replace('5+',5)
data['rcount'] = data['rcount'].astype('int64')
print(data['rcount'].dtype)

print(data.info())

#c. handling simple categorical data gender and facid
data['gender'] = data['gender'].map({'M':0, 'F':1})
data = pd.get_dummies(data, columns = ['facid'], prefix = 'facid', drop_first= True)

print(data.head())
print(data.info())

# prepare the data for modeling 
# y = lengthof stay
# x = ? 

y = data['lengthofstay']
x = data.drop(columns=['lengthofstay'])

print("shape of featured (x):", x.shape)
print("shape of target y :", y.shape)

#a. train-test split
# to accurately evaluate your mpdel's performance on unseen data, you must 
# split your data into two sets: one for training the model and one for testing it

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#b. model training(fitting)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

#train the model (find the best parameters - weights on each features)
model.fit(X_train, y_train)
print("Model training completed.")

#c. Inspecting the parameters 
# after training, you can directly inspect which features the model deemed most 
# relevant by looking at the magnitude(size) of the coefficients

# step 1: create a dataframe to easily view feature names and their coefficients 
feature_names = X_train.columns 
coefficients = model.coef_

# step 2: combine them into a single, sorted dataframe 
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("Intercept:", model.intercept_)
print("Coefficients:", coef_df.head(10))

#d. Model Evaluation  using y_pred and y_test 
from sklearn.metrics import mean_squared_error, r2_score
# 1. make predictions on the test set 
y_pred = model.predict(X_test)
#2. calculate R-squared (R2 score) 
r2 = r2_score(y_test, y_pred)
#3. calculate mean squared error(MSE) and then root mean squared error(RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"r-squared (R2) score: {r2:.3f}")
print(f"rooted mean squared error (rmse): {rmse:.3f}")

#e. Model deployment 
#1. save the model 
import pickle 
model_filename = 'length_of_stay_model.pkl'
#2. save the model object to the file 
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}.")

# 3. create a prediction function (API backend)
# needed to load the saved model and create a function that hanldes predictions 

# 1. Load the trained model
with open('length_of_stay_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# The list of feature names used during training (critical for matching input order!)
# NOTE: This list needs to match the X_train columns exactly, especially 'gender', 'rcount', and the facid dummies you kept.
# You will replace the features below with the exact list from your X_train DataFrame.
# Example features:
FEATURE_COLUMNS = [
    'vdate', 'rcount', 'gender', 'dialysisrenalendstage', 'asthma',
    'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor',
    'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
    'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
    'creatinine', 'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9',
    'facid_B', 'facid_C', 'facid_D', 'facid_E' # Assuming you dropped facid_A
]


def predict_los(input_data: dict) -> float:
    """
    Takes a dictionary of patient features and returns the predicted Length of Stay.
    """
    # 2. Convert the input dictionary into a DataFrame row
    input_df = pd.DataFrame([input_data])

    # 3. Ensure columns are in the correct order before prediction
    input_df = input_df[FEATURE_COLUMNS]

    # 4. Make the prediction
    prediction = loaded_model.predict(input_df)

    # 5. Return the single predicted value (in days)
    return float(prediction[0])

print(X_train.columns.tolist())

#4. connect to a web interface (streamlit)
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
