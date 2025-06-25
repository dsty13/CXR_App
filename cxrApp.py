import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import time

# === Load model dan komponen ===
scaler = joblib.load('STD_scaler.pkl')
pca = joblib.load('PCA_COMPONENT2.pkl')
model_non_pca = tf.keras.models.load_model('model_nonPCA.keras')
model_pca = tf.keras.models.load_model('PCA_MODEL2.keras')

# === Load ResNet50 untuk ekstraksi fitur ===
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# === Fungsi ekstraksi fitur dari gambar ===
def extract_features_from_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224), color_mode='rgb')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        st.error(f"Gagal mengekstrak fitur: {e}")
        return None

# === Fungsi prediksi dan format probabilitas ===
def predict(features, use_pca=False):
    try:
        start_time = time.time()
        features_scaled = scaler.transform([features])
        original_features_display = features_scaled.flatten()

        if use_pca:
            pca_features = pca.transform(features_scaled)
            prediction = model_pca.predict(pca_features, verbose=0)
            used_features = pca_features.flatten()
        else:
            prediction = model_non_pca.predict(features_scaled, verbose=0)
            used_features = original_features_display

        predicted_class = np.argmax(prediction, axis=1)[0]
        label_map = {0: "COVID-19", 1: "Pneumonia", 2: "Normal"}
        elapsed_time = time.time() - start_time

        # Format hasil probabilitas dalam persen
        probs = prediction.flatten()
        probs_percent = {label_map[i]: f"{probs[i]*100:.2f}%" for i in range(len(probs))}

        return label_map[predicted_class], original_features_display, used_features, elapsed_time, probs_percent
    except Exception as e:
        st.error(f"Gagal memproses prediksi: {e}")
        return None, None, None, None, None

# === Data sample ===
sample_images = {
    "COVID-19": ["images/covid19/covid1.png", "images/covid19/covid2.png", "images/covid19/covid3.png"],
    "Pneumonia": ["images/pneumonia/pneumonia1.png", "images/pneumonia/pneumonia2.png", "images/pneumonia/pneumonia3.png"],
    "Normal": ["images/normal/normal1.png", "images/normal/normal2.png", "images/normal/normal3.png"]
}

# === Streamlit Config dan Style ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        width: 100%;
        height: 50px;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Klasifikasi COVID-19 dan Pneumonia dari Citra X-Ray Dada")

# === Sidebar Navigasi ===
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Navigation", ["🏠 Home", "🖼️ Input image", "🧪 Try Sample"])

# === Beranda ===
if page == "🏠 Beranda":
    st.markdown("### Deskripsi Aplikasi:")
    st.markdown("""
    - Mengklasifikasikan citra X-Ray dada menjadi: **COVID-19**, **Pneumonia**, atau **Normal**
    - Fitur diekstrak menggunakan `ResNet50`
    - Klasifikasi menggunakan `Feed Forward Neural Network (FFNN)`
    - Bisa memilih klasifikasi **dengan atau tanpa PCA**
    - Akurasi model **Dengan PCA** mencapai 91%
    - Akurasi model **Tanpa PCA**  mencapai 88%
    """)
    if st.button("📥 Unduh Dataset dari Kaggle"):
        st.markdown("[Link Dataset COVIDQU - Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)", unsafe_allow_html=True)

# === Halaman Input Gambar ===
elif page == "🖼️ Input Gambar":
    st.subheader("Input Gambar dari Lokal")
    uploaded_file = st.file_uploader("Unggah gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    use_pca = st.radio("Gunakan PCA untuk Prediksi?", ["Ya", "Tidak"]) == "Ya"

    if uploaded_file:
        temp_path = "temp_image.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(temp_path, caption="Gambar yang Diunggah", width=400)

        if st.button("🔍 Prediksi", key="pred_lokal"):
            features = extract_features_from_image(temp_path)
            if features is not None:
                result, ori_feats, used_feats, pred_time, probs = predict(features, use_pca=use_pca)

                if ori_feats is not None:
                    with st.expander("📊 Fitur Asli (2048 dimensi)"):
                        st.write(ori_feats)

                if use_pca and used_feats is not None:
                    with st.expander(f"📉 Fitur Setelah PCA (jumlah: {len(used_feats)})"):
                        st.write(used_feats)

                if result:
                    st.success(f"✅ Hasil Prediksi: **{result}**")

                if probs:
                    st.markdown("### 📈 Probabilitas Kelas:")
                    for label, prob in probs.items():
                        st.markdown(f"**{label}** : {prob}")

                if pred_time is not None:
                    st.info(f"⏱️ Waktu Prediksi: **{pred_time:.4f} detik**")

# === Halaman Try Sample ===
elif page == "🧪 Try Sample":
    st.subheader("Coba Gambar Sampel")
    use_pca = st.radio("Gunakan PCA?", ["Ya", "Tidak"], key="pca_try_sample") == "Ya"

    col1, col2, col3 = st.columns(3)
    for cls, col in zip(["COVID-19", "Pneumonia", "Normal"], [col1, col2, col3]):
        with col:
            st.write(f"### {cls}")
            for img_path in sample_images[cls]:
                filename = os.path.basename(img_path)
                if st.button(f"Prediksi {filename}", key=img_path):
                    st.image(img_path, caption=filename, use_container_width=True)
                    features = extract_features_from_image(img_path)
                    if features is not None:
                        result, ori_feats, used_feats, pred_time, probs = predict(features, use_pca=use_pca)

                        if ori_feats is not None:
                            with st.expander("📊 Fitur Asli (2048 dimensi)"):
                                st.write(ori_feats)

                        if use_pca and used_feats is not None:
                            with st.expander(f"📉 Fitur Setelah PCA (jumlah: {len(used_feats)})"):
                                st.write(used_feats)

                        if result:
                            st.success(f"✅ Hasil Prediksi: **{result}**")

                        if probs:
                            st.markdown("### 📈 Probabilitas Kelas:")
                            for label, prob in probs.items():
                                st.markdown(f"**{label}** : {prob}")

                        if pred_time is not None:
                            st.info(f"⏱️ Waktu Prediksi: **{pred_time:.4f} detik**")
