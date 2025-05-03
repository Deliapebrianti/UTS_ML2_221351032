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
st.title("Prediksi Risiko Penyakit jantung")

# Input form
st.header("Informasi Pasein")
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoking = st.selectbox("Meroko", ["No", "Yes"])
alcohol = st.selectbox("Minum Alkohol", ["No", "Yes"])
stroke = st.selectbox("Riwayat Stroke", ["No", "Yes"])
physical_health = st.slider("Kesehatan Fisik (hari terganggu dalam 30 hari terakhir)", 0, 30, 0)
mental_health = st.slider("Kesehatan Mental (hari terganggu dalam 30 hari terakhir)", 0, 30, 0)
diff_walking = st.selectbox("Kesulitan Berjalan", ["No", "Yes"])
sex = st.selectbox("Jenis Kelamin", ["Female", "Male"])
age = st.selectbox("Kategori Umur", ["18-24", "25-29", "30-34", "35-39", "40-44", 
                                   "45-49", "50-54", "55-59", "60-64", "65-69", 
                                   "70-74", "75-79", "80 or older"])
race = st.selectbox("Ras/Etnis", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other"])
diabetic = st.selectbox("Diabetes", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
physical_activity = st.selectbox("Aktifitas Fisik", ["No", "Yes"])
gen_health = st.selectbox("Kesehatan Umum", ["Poor", "Fair", "Good", "Very good", "Excellent"])
sleep_time = st.slider("Waktu Tidur (JAM)", 1, 24, 7)
asthma = st.selectbox("Asma", ["No", "Yes"])
kidney_disease = st.selectbox("Penyakit Ginja", ["No", "Yes"])
skin_cancer = st.selectbox("Kanker Kulit", ["No", "Yes"])

if st.button("Prediksi penyakit Jantung"):
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
    risk = "Resiko Tinggi" if prediction > 0.5 else "Resiko Rendah"
    
    st.success(f"Prediction: {risk} ({probability:.2f}% probability)")
    
    if risk == "Resiko Tinggi":
        st.warning("Konsultasikan dengan tenaga medis untuk evaluasi lebih lanjut.")
        st.info("Pertahankan kebiasaan sehat untuk menjaga kesehatan jantung Anda!")
