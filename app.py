import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

st.write(""" 
# Student Performance Predictor
This app predicts if student is passed or Failed based on features your selecting from the sidebar

checkout the code at Github : https://github.com/madhuammulu8/Student-Performance-Predictor/tree/main
""")
df = pd.read_csv("student_data.csv")

outcomes = df['passed']
data = df.drop('passed', axis=1)

data = data.apply(lambda col: pd.factorize(col, sort=True)[0])

scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
scaled = scaler.transform(data)
train = pd.DataFrame(scaled, columns=data.columns)

X_train, X_test, y_train, y_test = train_test_split(train, outcomes, test_size=0.2, shuffle=False)

# Training
def random_forest():
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    return model

# Make predictions using selected features
def predict_pass(selected_features):
    # Encode input values
    input_data = pd.DataFrame(columns=data.columns)
    for feature in data.columns:
        if feature in selected_features:
            input_data[feature] = [selected_features[feature]]
        else:
            input_data[feature] = [0]  # Fill non-selected features with a dummy value

    input_data = input_data.apply(lambda col: pd.factorize(col, sort=True)[0])
    input_data = scaler.transform(input_data)

    # Make predictions
    model = random_forest()
    predictions = model.predict(input_data)

    return predictions[0]

gaurdian_selectbox = st.sidebar.selectbox("Select the Guardian Here ",["father","mother","other"],key="Gaurdian")
traveltime_selectbox= st.sidebar.selectbox("Select the Hour of Travel Time spent to School ",[1,2,3,4])
studttime_selectbox = st.sidebar.selectbox("Select the study TIme",[1,2,3,4],key="studytime")
internet_selectbox = st.sidebar.selectbox("Internet Availabilty ",["yes","no"],key="internet")
freetime_selectbox = st.sidebar.selectbox("Select the Free time student has ",[1,2,3,4,5],key="free time")
absences_selectbox =  st.sidebar.selectbox('Select the Absences',df['absences'].unique().tolist())

# Example usage
selected_features = {
    'guardian': gaurdian_selectbox,
    'traveltime': traveltime_selectbox,
    'studytime': studttime_selectbox,
    'internet': internet_selectbox,
    'freetime': freetime_selectbox,
    'absences':  absences_selectbox 
}

prediction = predict_pass(selected_features)
if prediction == "yes":
    st.success("The student has Passed based on the Features You have selected", icon="✅")
else:
    st.error("The student has Failed", icon="🚨")

