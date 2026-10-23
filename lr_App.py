import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('Linear_Regression_model.pkl', 'rb'))

# Lets Create web app
st.title("Scikit-learn linear regression model")

tv_budget = st.number_input("Enter Tv sales....", min_value=0.0, step=1.0)
Radio_budget = st.number_input("Enter Radio sales....", min_value=0.0, step=1.0)
Newspaper_budget = st.number_input("Enter Newspaper sales....", min_value=0.0, step=1.0)

if st.button("predict"):
    features = np.array([[tv_budget, Radio_budget, Newspaper_budget]], dtype=np.float64)
    result = model.predict(features).reshape(1, -1)
    st.write("Predict sale ::::", result[0])