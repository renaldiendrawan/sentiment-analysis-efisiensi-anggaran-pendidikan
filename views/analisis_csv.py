import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from utils import predict_sentiment

def cari_kolom_teks(daftar_kolom):
    kata_kunci = ['komentar', 'teks', 'tweet', 'text', 'opini', 'caption', 'review', 'isi']
    for i, col in enumerate(daftar_kolom):
        if any(kata in col.lower() for kata in kata_kunci):
            return i
    return 0 

def render_analisis_csv(model, tokenizer):
    st.title("📂 Analisis File CSV (Batch)")
    st.markdown("Unggah file data (Excel/CSV) yang berisi ribuan komentar, dan biarkan AI menganalisis sentimennya secara massal.")

    # 1. INISIALISASI SESSION STATE
    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = None
    if 'original_text_col' not in st.session_state:
        st.session_state['original_text_col'] = None

    # ==============================================================================
    # 2. AREA UPLOAD FILE
    # ==============================================================================
    uploaded_file = st.file_uploader("Upload File CSV di sini:", type=['csv'])
    
    if uploaded_file is None:
        st.session_state['batch_results'] = None
        st.session_state['original_text_col'] = None

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        
        st.markdown("---")
        st.subheader("⚙️ Konfigurasi Analisis")
        
        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            index_default = cari_kolom_teks(df_upload.columns)
            text_col = st.selectbox("Pilih Kolom yang Berisi Teks Opini:", df_upload.columns, index=index_default)
        with col_cfg2:
            st.info(f"📊 Total Data: **{len(df_upload)} baris**")

        if st.button("🚀 Mulai Proses Analisis", type="primary", use_container_width=True):
            with st.spinner('🤖 AI sedang memproses... Mohon tunggu.'):
                results_label, results_clean = [], []
                my_bar = st.progress(0, text="Memproses data...")
                total_data = len(df_upload)
                
                for i, row in df_upload.iterrows():
                    teks = str(row[text_col])
                    lbl, conf, _, cln = predict_sentiment(teks, model, tokenizer)
                    results_label.append(lbl) 
                    results_clean.append(cln)
                    
                    persen = (i + 1) / total_data
                    my_bar.progress(persen, text=f"Selesai: {i+1} dari {total_data} data ({int(persen*100)}%)")
                
                df_upload['Teks_Bersih'] = results_clean
                df_upload['Prediksi_Sentimen'] = results_label
                
                st.session_state['batch_results'] = df_upload
                st.session_state['original_text_col'] = text_col 
                st.success("✅ Analisis Berhasil Diselesaikan!")

    # ==============================================================================
    # 3. AREA HASIL PREDIKSI
    # ==============================================================================
    if st.session_state['batch_results'] is not None:
        st.markdown("---")
        df_final = st.session_state['batch_results'].copy()
        df_final.index = range(1, len(df_final) + 1)
        kolom_asli = st.session_state['original_text_col']

        df_final['Prediksi_Sentimen'] = df_final['Prediksi_Sentimen'].astype(str).str.strip().str.title()
        
        tab1, tab2, tab3 = st.tabs(["📋 Tabel Hasil", "📊 Statistik & Grafik", "☁️ WordCloud"])
        
        # --- TAB 1: TABEL HASIL ---
        with tab1:
            st.subheader("📋 Pratinjau Data Hasil Analisis")
            st.dataframe(df_final, use_container_width=True)
            
            st.write("")
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil Lengkap (CSV)", data=csv, file_name="Hasil_Analisis_Batch.csv", mime="text/csv")
        
        # --- TAB 2: STATISTIK & GRAFIK ---
        with tab2:
            st.subheader("📊 Statistik Sentimen Data Baru")
            count_res = df_final['Prediksi_Sentimen'].value_counts().reset_index()
            count_res.columns = ['Sentimen', 'Jumlah']
            
            warna_map = pd.DataFrame({
                'Sentimen': ['Positif', 'Netral', 'Negatif'],
                'Warna': ['#00CC96', '#808495', '#FF4B4B']
            })
            chart_data = count_res.merge(warna_map, on='Sentimen')

            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.caption("Distribusi Jumlah")
                c = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sentimen', sort=['Negatif', 'Netral', 'Positif']),
                    y='Jumlah',
                    color=alt.Color('Sentimen', scale=alt.Scale(domain=['Positif', 'Netral', 'Negatif'], range=['#00CC96', '#808495', '#FF4B4B']), legend=None),
                    tooltip=['Sentimen', 'Jumlah']
                ).properties(height=350)
                st.altair_chart(c, use_container_width=True)
                
            with col_stat2:
                st.caption("Proporsi Persentase")
                fig_pie = px.pie(count_res, names='Sentimen', values='Jumlah', hole=0.4, 
                                 color='Sentimen', color_discrete_map={'Negatif':'#FF4B4B', 'Netral':'#808495', 'Positif':'#00CC96'})
                st.plotly_chart(fig_pie, use_container_width=True)

        # --- TAB 3: WORDCLOUD ---
        with tab3:
            st.subheader("☁️ WordCloud: Representasi Visual Teks")
            
            # Dropdown 
            pilihan_wc = [
                "1. Data Mentah", 
                "2. Data Bersih (Preprocessed)", 
                "3. Sentimen NEGATIF", 
                "4. Sentimen NETRAL", 
                "5. Sentimen POSITIF"
            ]
            sent_choice = st.selectbox("Pilih Kategori Teks (Langsung Berubah):", pilihan_wc)

            filter_sentimen = df_final['Prediksi_Sentimen'].str.lower()
            text_wc = ""
            tema_warna = 'viridis'

            if "Mentah" in sent_choice:
                text_wc = " ".join(df_final[kolom_asli].astype(str))
                tema_warna = "cividis" 
            elif "Bersih" in sent_choice:
                text_wc = " ".join(df_final['Teks_Bersih'].astype(str))
                tema_warna = "viridis" 
            elif "NEGATIF" in sent_choice:
                text_wc = " ".join(df_final[filter_sentimen == 'negatif']['Teks_Bersih'].astype(str))
                tema_warna = "Reds" 
            elif "NETRAL" in sent_choice:
                text_wc = " ".join(df_final[filter_sentimen == 'netral']['Teks_Bersih'].astype(str))
                tema_warna = "Greys" 
            elif "POSITIF" in sent_choice:
                text_wc = " ".join(df_final[filter_sentimen == 'positif']['Teks_Bersih'].astype(str))
                tema_warna = "Greens" 
            
            # TAMPILKAN WORDCLOUD
            if not text_wc.strip():
                st.warning("⚠️ Tidak ada data untuk kategori ini di file Anda.")
            else:
                with st.spinner("Menggambar WordCloud..."):
                    wc = WordCloud(width=800, height=400, background_color='white', colormap=tema_warna, max_words=100).generate(text_wc)
                    wc_image = wc.to_image() 
                    wc_array = np.array(wc_image)
                    
                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc_array, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)