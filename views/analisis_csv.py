import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from utils import predict_sentiment

def render_analisis_csv(model, tokenizer):
    st.title("📂 Analisis File CSV (Batch)")
    st.markdown("Unggah file data (CSV) yang berisi ribuan komentar, dan biarkan AI menganalisis sentimennya secara massal.")
    
    st.info("💡 **Panduan Upload:** Pastikan file CSV Anda memiliki kolom bernama **Teks Tweet** yang berisi teks/opini. Jika namanya berbeda, mohon ubah terlebih dahulu di Excel.")

    # 1. INISIALISASI SESSION STATE
    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = None
    if 'original_text_col' not in st.session_state:
        st.session_state['original_text_col'] = None

    # ==============================================================================
    # 2. AREA UPLOAD FILE
    # ==============================================================================
    uploaded_file = st.file_uploader("Upload File CSV di sini:")
    
    if uploaded_file is None:
        st.session_state['batch_results'] = None
        st.session_state['original_text_col'] = None

    if uploaded_file is not None:
        # --- VALIDASI EKSTENSI (MEMENUHI TEST CASE 2) ---
        if not uploaded_file.name.lower().endswith('.csv'):
            st.error("❌ **Error:** Format file tidak didukung! Sistem hanya dapat memproses file berekstensi **.csv**.")
            return # Menghentikan proses agar tidak lanjut ke bawah
            
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            # --- VALIDASI 1: Cek apakah file kosong ---
            if df_upload.empty:
                st.error("❌ File CSV yang Anda unggah kosong (0 baris). Silakan periksa kembali file Anda.")
                return

            # --- VALIDASI 2: VALIDASI KOLOM KETAT (STRICT) ---
            KOLOM_WAJIB = "Teks Tweet"
            
            # Cek apakah kolom wajib ada (case-sensitive)
            if KOLOM_WAJIB not in df_upload.columns:
                st.error(f"❌ **Error Format:** File CSV Anda tidak memiliki kolom bernama **'{KOLOM_WAJIB}'**.")
                st.warning(f"Perbaiki file Anda: Buka di Excel, ubah nama kolom yang berisi teks opini menjadi '{KOLOM_WAJIB}', simpan kembali sebagai CSV, lalu unggah ulang.")
                return

            st.markdown("---")
            st.subheader("⚙️ Konfigurasi Analisis")
            
            text_col = KOLOM_WAJIB
            st.success(f"✅ Kolom target **'{text_col}'** ditemukan! Total Data: **{len(df_upload)} baris**.")

            if st.button("🚀 Mulai Proses Analisis", type="primary", use_container_width=True):
                with st.spinner('🤖 AI sedang memproses... Mohon tunggu.'):
                    # Membersihkan nilai NaN sebelum diproses
                    df_upload[text_col] = df_upload[text_col].fillna("")
                    
                    results_label, results_clean = [], []
                    my_bar = st.progress(0, text="Memproses data...")
                    total_data = len(df_upload)
                    error_count = 0
                    
                    for i, row in df_upload.iterrows():
                        teks = str(row[text_col])
                        
                        # Lewati jika teks kosong untuk mempercepat
                        if not teks.strip():
                            results_label.append("Netral")
                            results_clean.append("")
                        else:
                            try:
                                lbl, conf, _, cln = predict_sentiment(teks, model, tokenizer)
                                results_label.append(lbl) 
                                results_clean.append(cln)
                            except Exception as e:
                                results_label.append("Error")
                                results_clean.append("GAGAL DIPROSES")
                                error_count += 1
                        
                        persen = (i + 1) / total_data
                        my_bar.progress(persen, text=f"Selesai: {i+1} dari {total_data} data ({int(persen*100)}%)")
                    
                    # Simpan hasil ke DataFrame
                    df_upload['Teks_Bersih'] = results_clean
                    df_upload['Prediksi_Sentimen'] = results_label
                    
                    st.session_state['batch_results'] = df_upload
                    st.session_state['original_text_col'] = text_col 
                    
                    if error_count > 0:
                        st.warning(f"⚠️ Analisis selesai, namun ada **{error_count} baris yang gagal diproses** (ditandai dengan label 'Error').")
                    else:
                        st.success("✅ Semua data berhasil dianalisis tanpa masalah!")

        except pd.errors.EmptyDataError:
            st.error("❌ **Error:** File CSV kosong atau format rusak.")
        except pd.errors.ParserError:
            st.error("❌ **Error Parsing:** Susunan koma (delimiter) pada file CSV berantakan. Harap simpan ulang file Excel ke format CSV.")
        except Exception as e:
            st.error(f"❌ **Kesalahan Sistem:** Terjadi masalah yang tidak terduga: `{e}`")

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