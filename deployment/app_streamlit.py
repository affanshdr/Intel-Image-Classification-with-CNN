import streamlit as st
from PIL import Image

from config import CLASSES, IMG_SIZE
from inference import get_model, predict_from_image

st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🖼️",
    layout="centered",
)

st.markdown("""
<style>
    /* Mengimpor font modern */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Styling Judul Utama */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00F0FF 0%, #0088FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }

    /* Styling untuk Metric Score */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #00F0FF !important;
        text-shadow: 0px 0px 15px rgba(0, 240, 255, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #CBD5E1 !important;
    }

    /* Styling area Drag & Drop Uploader */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #0088FF !important;
        background-color: rgba(0, 136, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(0, 240, 255, 0.1) !important;
        border-color: #00F0FF !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 240, 255, 0.1);
    }
    
    /* Box Shadow untuk gambar input */
    [data-testid="stImage"] img {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Vision Classifier Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistem klasifikasi gambar cerdas dengan <b>CNN</b> untuk mengenali <b>Buildings</b>, <b>Forest</b>, dan <b>Mountain</b>.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Model:** `model1_sederhana.h5`")
    st.write(f"**Ukuran input:** {IMG_SIZE}×{IMG_SIZE} px")
    st.write("**Kelas:**")
    for cls in CLASSES:
        st.write(f"- {cls}")
    st.divider()
    
    st.header("Pengaturan Prediksi")
    threshold = st.slider(
        "Threshold Confidence (Batas Minimum)", 
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Jika confidence di bawah batas ini, gambar akan dikategorikan sebagai 'Lainnya / Tidak Diketahui'."
    )
    
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

        if confidence < threshold:
            st.warning("⚠️ Confidence rendah! Gambar ini sepertinya tidak termasuk ke dalam ketiga kategori yang ada.")
            st.error("**Label:** Lainnya / Tidak Diketahui")
        else:
            st.success(f"**Label:** {label}")
            
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        st.markdown("**Skor semua kelas:**")
        for cls, score in sorted(
            result["all_scores"].items(), key=lambda x: x[1], reverse=True
        ):
            st.progress(score, text=f"{cls}: {score * 100:.2f}%")
else:
    st.info("Silakan unggah gambar untuk memulai prediksi.")
