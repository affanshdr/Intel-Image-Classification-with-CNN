import streamlit as st
from PIL import Image

from config import CLASSES, IMG_SIZE
from inference import get_model, predict_from_image

st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🖼️",
    layout="centered",
)

st.title("Intel Image Classification")
st.markdown(
    "Klasifikasi gambar menggunakan **Model 1 Sederhana (CNN)** — "
    "mengenali *buildings*, *forest*, dan *mountain*."
)

with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Model:** `model1_sederhana.h5`")
    st.write(f"**Ukuran input:** {IMG_SIZE}×{IMG_SIZE} px")
    st.write("**Kelas:**")
    for cls in CLASSES:
        st.write(f"- {cls}")
    st.divider()
    st.caption("FastAPI: `http://localhost:8000` | Mobile web: `http://localhost:8000/mobile`")

@st.cache_resource
def load_cached_model():
    return get_model()

load_cached_model()

uploaded_file = st.file_uploader(
    "Unggah gambar untuk diklasifikasi",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Hasil Prediksi")
        with st.spinner("Memproses..."):
            result = predict_from_image(image)

        label = result["label"]
        confidence = result["confidence"]

        st.success(f"**Label:** {label}")
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        st.markdown("**Skor semua kelas:**")
        for cls, score in sorted(
            result["all_scores"].items(), key=lambda x: x[1], reverse=True
        ):
            st.progress(score, text=f"{cls}: {score * 100:.2f}%")
else:
    st.info("Silakan unggah gambar untuk memulai prediksi.")
