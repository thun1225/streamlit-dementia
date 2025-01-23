import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Dementia Prediction App
This app predicts the likelihood of dementia based on health indicators like diabetes, alcohol levels, heart rate, and more.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    diabetic = st.sidebar.slider('Diabetic (1: Yes, 0: No)', 0, 1, 0)
    alcohol_level = st.sidebar.slider('Alcohol Level (in %)', 0.0, 5.0, 0.5)
    heart_rate = st.sidebar.slider('Heart Rate (bpm)', 40, 150, 70)
    blood_oxygen_level = st.sidebar.slider('Blood Oxygen Level (%)', 80, 100, 95)
    body_temperature = st.sidebar.slider('Body Temperature (°C)', 35.0, 42.0, 37.0)
    weight = st.sidebar.slider('Weight (kg)', 30, 150, 70)
    mri_delay = st.sidebar.slider('MRI Delay (days)', 0, 365, 30)

    data = {
        'diabetic': diabetic,
        'alcohol_level': alcohol_level,
        'heart_rate': heart_rate,
        'blood_oxygen_level': blood_oxygen_level,
        'body_temperature': body_temperature,
        'weight': weight,
        'mri_delay': mri_delay,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Example dataset and model
# Replace this with your actual dementia dataset
X = [[0, 0.5, 70, 95, 37.0, 70, 30], [1, 1.0, 80, 90, 38.0, 80, 60]]
Y = [0, 1]  # 0: No Dementia, 1: Dementia

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
st.write('Dementia' if prediction[0] == 1 else 'No Dementia')

st.subheader('Prediction Probability')
st.write(prediction_proba)
