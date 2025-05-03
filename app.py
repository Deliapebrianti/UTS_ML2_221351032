import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Memuat model dan objek preprocessing
@st.cache(allow_output_mutation=True)
def load_objects():
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('target_encoder.pkl', 'rb') as f:
        le_target = pickle.load(f)
    
    # Memuat model TFLite
    interpreter = tf.lite.Interpreter(model_path="heart_disease_model.tflite")
    interpreter.allocate_tensors()
    
    return label_encoders, scaler, le_target, interpreter

label_encoders, scaler, le_target, interpreter = load_objects()

# Aplikasi Streamlit
st.title("Prediksi Risiko Penyakit Jantung")

# Formulir Input
st.header("Informasi Pasien")
bmi = st.slider("Indeks Massa Tubuh (BMI)", 10.0, 50.0, 25.0)
merokok = st.selectbox("Apakah Anda Merokok?", ["Tidak", "Ya"])
alkohol = st.selectbox("Apakah Anda Mengonsumsi Alkohol?", ["Tidak", "Ya"])
stroke = st.selectbox("Riwayat Stroke", ["Tidak", "Ya"])
kesehatan_fisik = st.slider("Kesehatan Fisik (hari terganggu dalam 30 hari terakhir)", 0, 30, 0)
kesehatan_mental = st.slider("Kesehatan Mental (hari terganggu dalam 30 hari terakhir)", 0, 30, 0)
sulit_berjalan = st.selectbox("Kesulitan Berjalan", ["Tidak", "Ya"])
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
usia = st.selectbox("Kategori Usia", ["18-24", "25-29", "30-34", "35-39", "40-44", 
                                   "45-49", "50-54", "55-59", "60-64", "65-69", 
                                   "70-74", "75-79", "80 atau lebih"])
ras = st.selectbox("Ras/Etnis", ["Kaukasia", "Afrika-Amerika", "Asia", "Pribumi Amerika", "Lainnya"])
diabetes = st.selectbox("Diabetes", ["Tidak", "Ya", "Batas diabetes", "Ya (saat hamil)"])
aktivitas_fisik = st.selectbox("Aktivitas Fisik", ["Tidak", "Ya"])
kesehatan_umum = st.selectbox("Kesehatan Umum", ["Buruk", "Cukup", "Baik", "Sangat baik", "Luar biasa"])
waktu_tidur = st.slider("Waktu Tidur (jam)", 1, 24, 7)
asma = st.selectbox("Asma", ["Tidak", "Ya"])
penyakit_ginjal = st.selectbox("Penyakit Ginjal", ["Tidak", "Ya"])
kanker_kulit = st.selectbox("Kanker Kulit", ["Tidak", "Ya"])

if st.button("Prediksi Risiko Penyakit Jantung"):
    # Mempersiapkan data input
    input_data = {
        'BMI': bmi,
        'Smoking': merokok,
        'AlcoholDrinking': alkohol,
        'Stroke': stroke,
        'PhysicalHealth': kesehatan_fisik,
        'MentalHealth': kesehatan_mental,
        'DiffWalking': sulit_berjalan,
        'Sex': jenis_kelamin,
        'AgeCategory': usia,
        'Race': ras,
        'Diabetic': diabetes,
        'PhysicalActivity': aktivitas_fisik,
        'GenHealth': kesehatan_umum,
        'SleepTime': waktu_tidur,
        'Asthma': asma,
        'KidneyDisease': penyakit_ginjal,
        'SkinCancer': kanker_kulit
    }
    
    # Preprocessing
    df = pd.DataFrame([input_data])
    for col in df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    scaled_data = scaler.transform(df)
    
    # Prediksi
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], scaled_data.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    probabilitas = prediction[0][0] * 100
    risiko = "Risiko Tinggi" if prediction > 0.5 else "Risiko Rendah"
    
    st.success(f"Hasil Prediksi: {risk} ({probabilitas:.2f}% probability)")
    
    if risiko == "Risiko Tinggi":
        st.warning("Konsultasikan dengan tenaga medis untuk evaluasi lebih lanjut.")
    else:
        st.info("Pertahankan kebiasaan sehat untuk menjaga kesehatan jantung Anda!")
