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
