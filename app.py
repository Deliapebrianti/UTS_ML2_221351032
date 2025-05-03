import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load the model and preprocessing objects
@st.cache(allow_output_mutation=True)
def load_objects():
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('target_encoder.pkl', 'rb') as f:
        le_target = pickle.load(f)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="heart_disease_model.tflite")
    interpreter.allocate_tensors()
    
    return label_encoders, scaler, le_target, interpreter

label_encoders, scaler, le_target, interpreter = load_objects()

# Streamlit app
st.title("Heart Disease Risk Prediction")

# Input form
st.header("Patient Information")
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Drinking", ["No", "Yes"])
stroke = st.selectbox("History of Stroke", ["No", "Yes"])
physical_health = st.slider("Physical Health (days affected in last 30)", 0, 30, 0)
mental_health = st.slider("Mental Health (days affected in last 30)", 0, 30, 0)
diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44", 
                                   "45-49", "50-54", "55-59", "60-64", "65-69", 
                                   "70-74", "75-79", "80 or older"])
race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other"])
diabetic = st.selectbox("Diabetic", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
gen_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
sleep_time = st.slider("Sleep Time (hours)", 1, 24, 7)
asthma = st.selectbox("Asthma", ["No", "Yes"])
kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])

if st.button("Predict Heart Disease Risk"):
    # Prepare input data
    input_data = {
        'BMI': bmi,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health,
        'SleepTime': sleep_time,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }
    
    # Preprocess
    df = pd.DataFrame([input_data])
    for col in df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    scaled_data = scaler.transform(df)
    
    # Predict
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], scaled_data.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    probability = prediction[0][0] * 100
    risk = "High Risk" if prediction > 0.5 else "Low Risk"
    
    st.success(f"Prediction: {risk} ({probability:.2f}% probability)")
    
    if risk == "High Risk":
        st.warning("Consult with a healthcare professional for further evaluation.")
    else:
        st.info("Maintain healthy habits to keep your heart healthy!")