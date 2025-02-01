import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logistic_model.pkl')

# Define the app title
st.title("Titanic Survival Prediction App")

# User inputs for features
st.sidebar.header("Input Features")

# Collect user inputs
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.sidebar.slider("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 30.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Preprocess the inputs
sex = 1 if sex == "Female" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create input feature array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_C, embarked_Q]])

# Make prediction
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Show results
    if prediction == 1:
        st.success(f"The passenger is predicted to survive with a probability of {probability:.2f}.")
    else:
        st.error(f"The passenger is predicted not to survive with a probability of {1 - probability:.2f}.")