import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  
from wordcloud import WordCloud
import os
import math

def render_visualisasi():
    st.title("📈 Dashboard Visualisasi Data")
    st.markdown("Analisis visual interaktif terhadap data opini publik terkait kebijakan anggaran pendidikan.")

    # ==============================================================================
    # 1. LOAD DATA UTAMA
    # ==============================================================================
    file_path = 'data/Data_Lengkap_Tokenisasi.csv'
    
    if not os.path.exists(file_path):
        st.error(f"❌ File dataset tidak ditemukan di: {file_path}")
        return

    # Load Data
    df = pd.read_csv(file_path)

    if 'Label' in df.columns:
        df['Label_Clean'] = df['Label'].astype(str).str.lower().str.strip()
    else:
        st.error("❌ Kolom 'Label' tidak ditemukan dalam CSV.")
        return

    if 'created_at' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['created_at']).dt.date
    elif 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.date
    else:
        st.warning("⚠️ Kolom tanggal tidak ditemukan. Grafik tren waktu mungkin tidak muncul.")

    # ==============================================================================
    # 2. VISUALISASI DISTRIBUSI SENTIMEN (PIE & BAR)
    # ==============================================================================
    st.subheader("📊 Distribusi & Polaritas Sentimen")
    
    col_pie, col_bar = st.columns([1, 1.5])

    # --- A. PIE CHART ---
    with col_pie:
        df_pie = df['Label_Clean'].value_counts().reset_index()
        df_pie.columns = ['Sentimen', 'Jumlah']
        
        fig_pie = px.pie(
            df_pie, 
            names='Sentimen', 
            values='Jumlah', 
            hole=0.4, 
            color='Sentimen',
            color_discrete_map={'negatif':'#FF4B4B', 'netral':'#808495', 'positif':'#00CC96'},
            title="Persentase Sentimen"
        )
        fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- B. TREN WAKTU ---
    with col_bar:
        if 'Tanggal' in df.columns:
            start_date = pd.to_datetime("2025-02-01").date()
            end_date = pd.to_datetime("2025-03-31").date()
            
            df_filtered = df[
                (df['Tanggal'] >= start_date) & 
                (df['Tanggal'] <= end_date)
            ]
         
            kolom_label = 'Label' if 'Label' in df_filtered.columns else 'Label_Clean'
            
            df_trend = df_filtered.groupby(['Tanggal', kolom_label]).size().reset_index(name='Jumlah')
            
            fig_trend = px.line(
                df_trend, 
                x='Tanggal', 
                y='Jumlah', 
                color=kolom_label, 
                markers=True,
                color_discrete_map={
                    'negatif':'#FF4B4B', 'netral':'#808495', 'positif':'#00CC96', 
                    'Negatif':'#FF4B4B', 'Netral':'#808495', 'Positif':'#00CC96', 
                    'negative':'#FF4B4B', 'neutral':'#808495', 'positive':'#00CC96' 
                },
                title="Tren Sentimen Harian (Feb - Mar 2025)"
            )
            
            fig_trend.update_xaxes(range=[start_date, end_date])
            fig_trend.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Tweet", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Data Tanggal tidak tersedia untuk menampilkan tren.")

    # ==============================================================================
    # 3. WORDCLOUD
    # ==============================================================================
    st.subheader("☁️ WordCloud: Representasi Visual Teks")
    st.write("Kata-kata yang paling sering muncul dalam setiap kategori.")

    def generate_wc(text, colormap):
        if not isinstance(text, str) or not text.strip():
            st.warning("⚠️ Tidak ada data teks yang cukup.")
            return
        
        with st.spinner("Sedang menggambar WordCloud..."):
            try:
                wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap, max_words=100).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error WordCloud: {e}")

    # Tabs Navigasi
    tab_mentah, tab_bersih, tab_neg, tab_net, tab_pos = st.tabs([
        "Data Mentah", "Data Bersih", "Negatif", "Netral", "Positif"
    ])

    with tab_mentah:
        st.caption("Data dari kolom 'Teks Tweet' (Original)")
        generate_wc(" ".join(df['Teks Tweet'].astype(str)), 'cividis')

    with tab_bersih:
        st.caption("Data dari kolom 'Tweet_Final' (Preprocessed)")
        if 'Tweet_Final' in df.columns:
            generate_wc(" ".join(df['Tweet_Final'].astype(str)), 'viridis')
        else: st.warning("Kolom Tweet_Final tidak ada.")

    with tab_neg:
        st.caption("Kata dominan sentimen NEGATIF")
        subset = df[df['Label_Clean'] == 'negatif']
        if 'Tweet_Final' in df.columns: generate_wc(" ".join(subset['Tweet_Final'].astype(str)), 'Reds')

    with tab_net:
        st.caption("Kata dominan sentimen NETRAL")
        subset = df[df['Label_Clean'] == 'netral']
        if 'Tweet_Final' in df.columns: generate_wc(" ".join(subset['Tweet_Final'].astype(str)), 'Greys')

    with tab_pos:
        st.caption("Kata dominan sentimen POSITIF")
        subset = df[df['Label_Clean'] == 'positif']
        if 'Tweet_Final' in df.columns: generate_wc(" ".join(subset['Tweet_Final'].astype(str)), 'Greens')

    st.markdown("---")

    # ==============================================================================
    # 4. TOPIC MODELING 
    # ==============================================================================
    st.subheader("📌 5. Topic Modeling (LDA) & Kata Kunci")
    st.write("Ekstraksi topik dominan dari hasil algoritma Latent Dirichlet Allocation (LDA).")

    path_lda = 'model/Hasil_Analisis_Topik_LDA.csv'
    if not os.path.exists(path_lda): path_lda = 'Hasil_Analisis_Topik_LDA.csv' 

    if os.path.exists(path_lda):
        try:
            df_lda = pd.read_csv(path_lda)
            
            def parse_lda_string(text_data):
                data_items = []
                for item in str(text_data).split(', '):
                    try:
                        weight, word = item.split('*')
                        data_items.append({'Kata': word.strip(), 'Bobot': float(weight)})
                    except: continue
                return pd.DataFrame(data_items).sort_values(by='Bobot', ascending=True)
            
            t_neg, t_net, t_pos = st.tabs(["Topik Negatif", "Topik Netral", "Topik Positif"])
            mapping = {'negatif': t_neg, 'netral': t_net, 'positif': t_pos}

            for sentimen, tab in mapping.items():
                with tab:
                    df_subset = df_lda[df_lda['Sentimen'] == sentimen]
                    
                    if df_subset.empty:
                        st.warning(f"Belum ada data topik untuk {sentimen}.")
                    else:
                        for idx, row in df_subset.iterrows():
                            topik_ke = row['Topik Ke']
                            df_chart = parse_lda_string(row['Kata Kunci (Bobot)'])
                            
                            if not df_chart.empty:
                                fig = px.bar(
                                    df_chart, x='Bobot', y='Kata', orientation='h',
                                    title=f"<b>Topik {topik_ke}:</b> Kata Kunci Dominan",
                                    color='Bobot',
                                    color_continuous_scale='Reds' if sentimen == 'negatif' else 'Greys' if sentimen == 'netral' else 'Greens'
                                )
                                fig.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                st.divider()
        except Exception as e:
            st.error(f"Gagal memproses data LDA: {e}")
    else:
        st.warning("⚠️ File 'Hasil_Analisis_Topik_LDA.csv' belum tersedia di folder model.")

    # ==============================================================================
    # 5. DATA EXPLORER & EVALUASI MODEL
    # ==============================================================================
    st.subheader("🔍 Data Explorer & Evaluasi Model")
    
    tab_data, tab_eval = st.tabs(["Data Explorer", "Tabel Performa (Evaluasi)"])

    # --- TAB 1: DATA EXPLORER ---
    with tab_data:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1: filter_label = st.selectbox("Filter Sentimen:", ['Semua', 'negatif', 'netral', 'positif'])
        with col_f2: search_keyword = st.text_input("Cari Tweet:", "")

        cols_available = [c for c in ['created_at', 'username', 'Teks Tweet', 'Label_Clean'] if c in df.columns]
        df_show = df[cols_available].copy()
        
        rename_map = {'created_at': 'Tanggal', 'username': 'Username', 'Label_Clean': 'Label'}
        df_show = df_show.rename(columns=rename_map)

        # Logika Filter
        if filter_label != 'Semua' and 'Label' in df_show.columns:
            df_show = df_show[df_show['Label'] == filter_label]
        
        if search_keyword and 'Teks Tweet' in df_show.columns:
            df_show = df_show[df_show['Teks Tweet'].str.contains(search_keyword, case=False, na=False)]

        df_show.index = range(1, len(df_show) + 1)
        
        # --- PAGINATION SYSTEM ---
        baris_per_halaman = 20
        total_data = len(df_show)
        total_halaman = math.ceil(total_data / baris_per_halaman)

        if total_data > 0:
            c_nav, c_stat = st.columns([1, 3])
            with c_nav:
                halaman = st.number_input("Halaman", min_value=1, max_value=max(1, total_halaman), step=1)
            with c_stat:
                st.write("") 
                st.caption(f"Menampilkan **{total_data}** Data (Halaman {halaman} dari {total_halaman})")

            # Slicing Data sesuai halaman
            start_idx = (halaman - 1) * baris_per_halaman
            end_idx = start_idx + baris_per_halaman
            df_page = df_show.iloc[start_idx:end_idx]

            st.dataframe(df_page, use_container_width=True)
        else:
            st.warning("Data tidak ditemukan.")

    # --- TAB 2: TABEL EVALUASI & CONFUSION MATRIX ---
    with tab_eval:
        # A. TABEL PERFORMA (Classification Report)
        st.subheader("1. Tabel Performa (Classification Report)")
        st.markdown("""
        Metrik evaluasi model berdasarkan data testing (20%):
        * **Precision**: Ketepatan tebakan.
        * **Recall**: Kemampuan menemukan data yang relevan.
        * **F1-Score**: Rata-rata harmonis (Paling penting untuk data tidak seimbang).
        """)
     
        path_perf = 'model/Tabel_Performa_LSTM.csv'
        if not os.path.exists(path_perf): path_perf = 'Tabel_Performa_LSTM.csv'

        if os.path.exists(path_perf):
            try:
                df_perf = pd.read_csv(path_perf, index_col=0)
                st.dataframe(
                    df_perf.style.highlight_max(axis=0, props='background-color: #FFEB3B; color: black; font-weight: bold'),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Gagal memuat tabel evaluasi: {e}")
        else:
            st.warning("⚠️ File 'Tabel_Performa_LSTM.csv' belum tersedia.")

        st.divider()

        # B. CONFUSION MATRIX (HEATMAP)
        st.subheader("2. Confusion Matrix")
        st.markdown("Visualisasi ini menunjukkan **detail kesalahan prediksi**. Sumbu Y adalah Label Asli, Sumbu X adalah Prediksi Model.")

        path_cm = 'model/Data_Confusion_Matrix.csv'
        if not os.path.exists(path_cm): path_cm = 'Data_Confusion_Matrix.csv'

        if os.path.exists(path_cm):
            try:
                df_cm_data = pd.read_csv(path_cm)
                
                if 'y_true' in df_cm_data.columns and 'y_pred' in df_cm_data.columns:
                    from sklearn.metrics import confusion_matrix
                    
                    labels = ['Negatif', 'Netral', 'Positif'] 
                    cm = confusion_matrix(df_cm_data['y_true'], df_cm_data['y_pred'])
                    
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        labels=dict(x="Prediksi Model", y="Label Aktual (Asli)", color="Jumlah Data"),
                        x=labels, 
                        y=labels,
                        color_continuous_scale='Blues',
                        aspect="auto"
                    )
                    fig_cm.update_layout(title="Confusion Matrix Heatmap")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    total_benar = np.trace(cm)
                    total_data = np.sum(cm)
                    akurasi_cm = (total_benar / total_data) * 100
                    st.caption(f"💡 **Interpretasi:** Dari total **{total_data}** data testing, model berhasil menebak benar sebanyak **{total_benar}** data ({akurasi_cm:.2f}%).")
                    
                else:
                    st.error("Format CSV Confusion Matrix salah. Harus ada kolom 'y_true' dan 'y_pred'.")
            except Exception as e:
                st.error(f"Gagal memproses Confusion Matrix: {e}")
        else:
            st.info("ℹ️ **Data Confusion Matrix belum tersedia.** Silakan jalankan kode penyimpanan `Data_Confusion_Matrix.csv` di Google Colab (Bagian Evaluasi).")