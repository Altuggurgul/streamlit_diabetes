import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.write("""
# Simple Diabetes Prediction App

""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0.1, 20.0, 2.0)
    Glucose = st.sidebar.slider('Glucose', 1.0, 200.0, 117.0)
    BloodPressure = st.sidebar.slider('BloodPressure', 10.0, 150.0, 60.0)
    SkinThickness = st.sidebar.slider('SkinThickness', 0.1, 99.0, 40.0)
    Insulin = st.sidebar.slider('Insulin', 2.0, 846.0, 100.0)
    BMI = st.sidebar.slider('BMI', 1.0, 35.0, 70.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.1, 998.0, 200.0)
    Age = st.sidebar.slider('Age', 10.0, 80.0, 35.0)
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

sheet_url = 'https://docs.google.com/spreadsheets/d/1OMurqWKO_E6IJHzX9HfvMssccJWpCspnA_B0KgLguPM/edit#gid=17834267'
csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
diabetes = pd.read_csv(csv_export_url)


#############
zero_columns = [col for col in diabetes.columns if (diabetes[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_columns:
    diabetes[col] = np.where(diabetes[col] == 0, np.nan, diabetes[col])

for col in zero_columns:
    diabetes.loc[diabetes[col].isnull(),col] = diabetes[col].mean()

#############
diabetes["Outcome_name"]= diabetes["Outcome"].apply(lambda x: "Diabet" if x==1 else "Non-Diabet")
diabetes.head()
#############


X = diabetes.drop(["Outcome","Outcome_name"], axis=1)
Y = diabetes.Outcome

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


diabets_isim= np.array(["Non-Diabetes","Diabetes"], dtype='<U10')


st.subheader('Class labels and their corresponding index number')
st.write(diabets_isim)

st.subheader('Prediction')
st.write(diabets_isim[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)



