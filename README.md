# Vision Classifier Pro (Intel Image Classification with CNN)

## Dibuat Oleh
- **Nama:** Affan Suhendar | 2308107010003
- **Nama:** Amirul Mirdas | 2308107010070


## 📌 Deskripsi Proyek
Proyek ini adalah sistem klasifikasi gambar cerdas yang dibangun menggunakan algoritma **Convolutional Neural Network (CNN)**. Model AI ini dilatih untuk mengenali dan membedakan tiga kategori citra lanskap/pemandangan:
- 🏢 **Buildings** (Bangunan/Gedung)
- 🌲 **Forest** (Hutan)
- ⛰️ **Mountain** (Gunung)

### ✨ Fitur Utama:
1. **Antarmuka Premium (Web GUI):** Dilengkapi dengan *front-end* interaktif berbasis **Streamlit** yang didesain modern (*dark mode*, efek *glowing*, animasi interaktif, dan tipografi elegan).
2. **REST API Endpoint:** Menyediakan layanan *backend* menggunakan **FastAPI** untuk kemudahan integrasi dengan aplikasi lain (seperti aplikasi *mobile* atau *web* eksternal).
3. **Smart Thresholding (Out-of-Distribution Detection):** Memiliki fitur penggeser *Confidence Threshold* yang memungkinkan sistem untuk "menolak" gambar di luar kategori (misalnya gambar hewan, makanan, dll.). Gambar yang tidak sesuai akan diklasifikasikan sebagai **"Lainnya / Tidak Diketahui"** secara otomatis.
4. **Robust Model Loading:** Menggunakan metode pembungkusan Keras *Custom Objects* untuk memastikan model (seperti layer `Dense`) dapat dimuat dengan aman terlepas dari konflik pembaruan versi TensorFlow (misal: perbaikan isu `quantization_config`).

---

## 🚀 Instruksi Instalasi & Penerapan (Deployment)

Ikuti langkah-langkah di bawah ini untuk menginstal dependensi dan menjalankan program di komputer (lokal) atau server Anda.

### 1. Persiapan Lingkungan (*Environment Setup*)
Sangat disarankan untuk menggunakan *Virtual Environment* (Lingkungan Virtual) agar paket-paket aplikasi ini terisolasi dengan baik.

```bash
# Clone repositori (jika belum)
git clone https://github.com/affanshdr/Intel-Image-Classification-with-CNN.git
cd Intel-Image-Classification-with-CNN

# Masuk ke direktori deployment
cd deployment

# Membuat Virtual Environment (opsional namun disarankan)
python3 -m venv venv

# Mengaktifkan Virtual Environment
# Untuk Linux/Mac:
source venv/bin/activate
# Untuk Windows:
venv\Scripts\activate
```

### 2. Instalasi Kebutuhan (Dependencies)
Pastikan lingkungan virtual sudah aktif, kemudian instal paket menggunakan `pip`:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Proyek ini telah dilengkapi dengan skrip `run.sh` untuk mempermudah eksekusi program. Pastikan Anda masih berada di dalam folder `deployment`.

#### Opsi A: Menjalankan Antarmuka Web (Streamlit)
Opsi ini sangat cocok jika Anda ingin menguji model secara visual lewat *browser* PC Anda.

```bash
# Memberikan izin eksekusi pada script (khusus Linux/Mac)
chmod +x run.sh

# Menjalankan antarmuka Streamlit
./run.sh streamlit
```
Aplikasi secara otomatis akan terbuka di peramban web pada alamat: **`http://localhost:8501`**

#### Opsi B: Menjalankan REST API (FastAPI)
Gunakan opsi ini jika Anda bertindak sebagai *backend* dan hanya ingin membuka akses *endpoint* prediksi (*inference*).

```bash
./run.sh api
```
- API Service berjalan pada: **`http://localhost:8000`**
- Interaktif Dokumentasi API (Swagger): **`http://localhost:8000/docs`**
- Halaman Web Mobile Sederhana: **`http://localhost:8000/mobile`**

---
*Dibuat untuk keperluan Proyek Ujian Akhir Semester (UAS) - Machine Learning.*
