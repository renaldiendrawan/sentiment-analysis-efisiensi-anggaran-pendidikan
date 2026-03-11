import streamlit as st
import pandas as pd
import os

def render_beranda():
    st.title("🎓 Selamat Datang di Sistem Analisis Sentimen")
    st.markdown("### Kebijakan Efisiensi Anggaran Pendidikan (2025)")
    st.markdown("Sistem ini dikembangkan untuk menganalisis opini masyarakat di media sosial X (Twitter) terkait isu pemotongan atau efisiensi anggaran di sektor pendidikan, seperti Dana BOS, PIP, KIP Kuliah, dan Tunjangan Guru.")

    st.markdown("---")

    # ==============================================================================
    # PENGAMBILAN DATA
    # ==============================================================================
    # 1. Akurasi Testing 
    try:
        df_perf = pd.read_csv('model/Tabel_Performa_LSTM.csv', index_col=0)
        akurasi_testing = round(df_perf.loc['accuracy', 'f1-score'] * 100, 2)
    except:
        akurasi_testing = 0.0 

    # --- 1. RINGKASAN METRIK MODEL ---
    st.subheader("📊 Ringkasan Model Machine Learning")
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("Arsitektur", "LSTM", "Deep Learning")
    m2.metric("Akurasi Model", f"{akurasi_testing}%", "Data Testing P5") 
    m3.metric("Pembagian Data", "80 : 20", "Latih : Uji")
    m4.metric("Metode Ekstraksi", "LDA", "Topic Modeling")

    st.markdown("---")

    # --- 2. METODOLOGI PENELITIAN ---
    st.subheader("🛠️ Metodologi & Arsitektur Sistem")
    col_metode1, col_metode2 = st.columns([1, 1])

    with col_metode1:
        st.markdown("""
        **Tahapan Pemrosesan:**
        1. **Crawling Data:** Pengambilan data via Tweet Harvest (Feb-Mar 2025).
        2. **Preprocessing:** Case folding, Cleaning, Tokenizing dan Normalisasi Slang. *(Tanpa Stopword & Stemming agar urutan konteks kalimat tetap utuh)*.
        3. **Word Embedding:** Standard Keras Embedding (Dimensi 128) dengan fitur *Masking*.
        4. **Deep Learning:** Model **Long Short-Term Memory (LSTM)** biasa untuk klasifikasi sentimen (Negatif, Netral, Positif).
        5. **Topic Modeling:** Latent Dirichlet Allocation (LDA) untuk mengetahui topik dominan.
        """)

    with col_metode2:
        st.info("**Mengapa menggunakan LSTM?** \n\nPenggunaan algoritma LSTM yang dipadukan dengan *Keras Embedding* terbukti lebih ringan dari segi komputasi namun tetap optimal dalam menangkap pola konteks kalimat secara sekuensial (berurutan). Fitur *Masking* memastikan padding kalimat tidak merusak makna sentimen.")
        
        st.success(f"**Hasil Pelatihan Model:** \nMelalui 5 tahapan percobaan (skenario 20% hingga 100% data latih), Akurasi Testing pada skenario P5 (100% data) mencapai **{akurasi_testing}%**. Ini menunjukkan model mampu memprediksi data baru dengan sangat baik.")

    st.markdown("---")

    # --- 3. FITUR SISTEM ---
    st.subheader("✨ Fitur Utama Sistem")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.success("**1. Dashboard Visualisasi**\n\nMenampilkan tren waktu, WordCloud, dan distribusi sentimen masyarakat secara interaktif.")
    with f2:
        st.warning("**2. Analisis Teks Langsung**\n\nPengguna dapat mengetikkan kalimat opini baru dan model akan memprediksi sentimennya secara *real-time*.")
    with f3:
        st.info("**3. Analisis File CSV**\n\nMengunggah data komentar/tweet dalam jumlah banyak sekaligus untuk dianalisis massal.")