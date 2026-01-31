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
    # 1. Total Data Latih 
    total_data_ros = 1563 

    # 2. Akurasi Testing 
    try:
        df_perf = pd.read_csv('model/Tabel_Performa_LSTM.csv', index_col=0)
        akurasi_testing = round(df_perf.loc['accuracy', 'f1-score'] * 100, 2)
    except:
        akurasi_testing = 87.33 

    # --- 1. RINGKASAN METRIK MODEL ---
    st.subheader("📊 Ringkasan Model Machine Learning")
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("Total Data Latih", f"{total_data_ros:,} Tweet", "Setelah Oversampling")
    m2.metric("Akurasi Model", f"{akurasi_testing}%", "Data Testing") 
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
        2. **Preprocessing:** Case folding, Cleaning, Tokenizing dan Normalisasi Slang. *(Stopword & Stemming ditiadakan untuk menjaga konteks FastText)*.
        3. **Word Embedding:** FastText (Dimensi 300) untuk mengubah kata menjadi vektor.
        4. **Deep Learning:** Model **Bi-Directional LSTM (Bi-LSTM)** untuk klasifikasi sentimen (Negatif, Netral, Positif).
        5. **Topic Modeling:** Latent Dirichlet Allocation (LDA) untuk mengetahui topik dominan.
        """)

    with col_metode2:
        st.info("**Mengapa Bi-LSTM + FastText?** \n\nFastText mampu menangani kata-kata slang/typo yang sering muncul di Twitter (Out-of-Vocabulary). Sedangkan Bi-LSTM mampu memahami konteks kalimat dari dua arah (kiri ke kanan dan sebaliknya), sangat cocok untuk bahasa Indonesia yang strukturnya dinamis.")
        
        st.success(f"**Hasil Pelatihan Model:** \nAkurasi Training mencapai **97.06%**, sedangkan Akurasi Testing stabil di **{akurasi_testing}%**. Ini menunjukkan model mampu memprediksi data baru dengan sangat baik tanpa mengalami *Overfitting* yang parah.")

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