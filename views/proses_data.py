import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
from sklearn.model_selection import train_test_split
import graphviz
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        return pd.DataFrame()

def render_proses_data():
    st.title("⚙️ Tahapan Proses Data & Modeling")
    st.markdown("Berikut adalah dokumentasi teknis alur pengolahan data dari mentah hingga evaluasi model, disertai penjelasan metodologi.")
    
    # LOAD DATA
    df_mentah = load_data('data/Data_Lengkap_Tokenisasi.csv') 

    # ==============================================================================
    # NAVIGASI 
    # ==============================================================================
    opsi_tahapan = [
        "1. Crawling Data", 
        "2. Preprocessing", 
        "3. Persiapan Data Latih", 
        "4. Arsitektur Model", 
        "5. Evaluasi Model",
        "6. Topic Modeling (LDA)"
    ]
    
    pilihan = st.radio("Pilih Tahapan Proses:", options=opsi_tahapan, horizontal=True, label_visibility="collapsed")
    
    st.markdown("---") 

    # ==============================================================================
    # KONTEN TAHAPAN
    # ==============================================================================
    
    # --- 1. CRAWLING DATA ---
    if pilihan == "1. Crawling Data":
        st.header("1. Pengumpulan Data (Crawling)")
        st.info("Tools: **Tweet-Harvest (Node.js)** API Scraper")
        
        st.success(f"✅ Total Data Terkumpul: **{len(df_mentah):,} Data** (Setelah Deduplikasi)")
        st.warning("⚠️ **Catatan Imbalance:** Distribusi sentimen awal tidak seimbang, ditangani dengan ROS (Random Over Sampling) pada tahap Training.")

        st.markdown("### 📋 Kriteria Pengambilan Data")
        st.markdown("""
        - **Platform**: X (Twitter)
        - **Periode**: 01 Februari 2025 - 31 Maret 2025
        - **Filter Sistem**: Hanya Bahasa Indonesia (`lang:id`) & Mengabaikan Retweet (`-is:retweet`).
        
        **Kata Kunci (Search Queries):**
        
        **1. Core Keywords (Isu Utama):**
        * `"efisiensi anggaran pendidikan" lang:id -is:retweet`
        * `"pemotongan anggaran pendidikan" lang:id -is:retweet`
        * `"anggaran pendidikan dikurangi" lang:id -is:retweet`

        **2. Program Spesifik:**
        * `("dana BOS" OR "bantuan operasional sekolah") ("dipotong" OR "dikurangi" OR "efisiensi" OR "kurang") lang:id -is:retweet`
        * `("PIP" OR "program indonesia pintar") ("dipotong" OR "dikurangi" OR "efisiensi" OR "cair") lang:id -is:retweet`
        * `("KIP Kuliah" OR "kartu indonesia pintar") ("dipotong" OR "dikurangi" OR "efisiensi" OR "sulit") lang:id -is:retweet`
        * `("tunjangan guru" OR "sertifikasi guru") ("dipotong" OR "dikurangi" OR "efisiensi" OR "telat") lang:id -is:retweet`

        **3. Kombinasi Isu Umum:**
        * `(anggaran OR dana) (pendidikan OR sekolah OR kampus OR guru) (efisiensi OR potong OR dikurangi OR berkurang) lang:id -is:retweet`
        
        - **Proses Lanjutan**: Deduplikasi (Hapus ID & Teks yang berulang).
        """)

        st.markdown("### 🔍 Preview Data Mentah")
        if not df_mentah.empty:
            search_mentah = st.text_input("Cari kata dalam Tweet (Mentah):", placeholder="Contoh: dana bos", key="cari_mentah")
            
            if search_mentah:
                df_tampil = df_mentah[df_mentah['Teks Tweet'].str.contains(search_mentah, case=False, na=False)].copy()
            else:
                df_tampil = df_mentah.copy()
            
            df_tampil = df_tampil[['created_at', 'username', 'Teks Tweet']].rename(columns={'created_at': 'Created At', 'username': 'Username'})
            df_tampil.index = range(1, len(df_tampil) + 1)
            
            st.dataframe(df_tampil, use_container_width=True, height=250)

    # --- 2. PREPROCESSING ---
    elif pilihan == "2. Preprocessing":
        st.header("2. Preprocessing Teks")
        
        st.markdown("""
        **Tujuan:** Mengubah data teks tidak terstruktur menjadi format bersih yang siap diproses mesin.
        
        Pada penelitian ini, kami memutuskan untuk **TIDAK MELAKUKAN Stemming & Stopword Removal**.
        * **Alasan:** Model Deep Learning (seperti LSTM/FastText) membutuhkan konteks kalimat utuh untuk memahami nuansa sentimen (contoh: kata *"tidak"* sangat penting untuk membalikkan makna *"suka"* menjadi *"tidak suka"*).
        """)

        with st.expander("ℹ️ Rincian 4 Langkah Preprocessing", expanded=True):
            st.markdown("""
            1.  **Case Folding:** Menyeragamkan huruf menjadi kecil (*lowercase*).
            2.  **Cleaning:** Menghapus elemen non-teks (URL, Mention `@`, Hashtag `#`, Angka, Emoji).
            3.  **Tokenizing:** Memecah kalimat menjadi potongan kata per kata.
            4.  **Normalisasi Slang:** Mengubah kata tidak baku (*bgt, gk, sy*) menjadi baku (*banget, tidak, saya*) menggunakan kamus *lexicon*.
            """)

        st.subheader("🔍 Komparasi Sebelum vs Sesudah")
        if not df_mentah.empty:
            search_pre = st.text_input("Cari kata (Hasil Akhir):", placeholder="Contoh: guru", key="cari_pre")
            
            cols = ['Teks Tweet', 'Tweet_CaseFolded', 'Tweet_Cleaned', 'Tweet_Tokenized', 'Tweet_Normalized', 'Tweet_Final']
            
            cols_exist = [c for c in cols if c in df_mentah.columns]
            df_tampil_pre = df_mentah[cols_exist].copy()

            if search_pre:
                df_tampil_pre = df_tampil_pre[df_tampil_pre['Tweet_Final'].str.contains(search_pre, case=False, na=False)]

            df_tampil_pre.index = range(1, len(df_tampil_pre) + 1)
            st.dataframe(df_tampil_pre, use_container_width=True, height=400)
        else:
            st.warning("Data preprocessing belum tersedia.")

    # --- 3. PERSIAPAN DATA LATIH ---
    elif pilihan == "3. Persiapan Data Latih":
        st.header("3. Transformasi & Splitting Data")
        
        st.markdown("""
        Agar teks dapat diproses oleh Neural Network, data harus diubah menjadi bentuk numerik (vektor).
        Selain itu, dilakukan penyeimbangan data agar model tidak bias.
        """)

        st.subheader("A. Tokenization & Padding")
        st.write("Setiap kata unik dalam dataset diberi ID angka. Karena panjang tweet berbeda-beda, kita lakukan **Padding** agar semua input memiliki panjang seragam (**100 kata**).")

        if not df_mentah.empty and 'Label' in df_mentah.columns:
            df_token = df_mentah.dropna(subset=['Label']).copy()
            
            # Helper simulasi token
            def get_word_id(word): return int(hashlib.md5(word.encode()).hexdigest(), 16) % 3000 + 1
            
            df_token['Detail Token'] = df_token['Tweet_Final'].apply(lambda t: ", ".join([f"{w}:{get_word_id(w)}" for w in str(t).split()[:10]]))
            df_token['Padding Sequence (100)'] = df_token['Tweet_Final'].apply(lambda t: str(([get_word_id(w) for w in str(t).split()] + [0]*100)[:20]) + " ...")
            
            st.dataframe(df_token[['Tweet_Final', 'Detail Token', 'Padding Sequence (100)']], use_container_width=True)

            st.markdown("---")
            st.subheader("B. Splitting & ROS (Random Over Sampling)")
            
            st.markdown("""
            **Masalah:** Data sentimen seringkali tidak seimbang (misal: Negatif 1000, Positif cuma 200).
            **Solusi (ROS):** Kami menduplikasi data minoritas (Positif/Netral) secara acak hingga jumlahnya setara dengan kelas mayoritas (Negatif).
            """)

            df_train, df_test = train_test_split(df_token, test_size=0.2, random_state=42, stratify=df_token['Label'])
            kelas_mayoritas = df_train['Label'].value_counts().max()
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Data Latih (80%)", f"{len(df_train):,} Sample", "Untuk melatih model")
            col_metric2.metric("Data Uji (20%)", f"{len(df_test):,} Sample", "Untuk validasi akhir")
            col_metric3.metric("Target ROS", f"{kelas_mayoritas}", "Per Kelas Sentimen")
            
            st.success(f"✅ **Status Data:** Dataset latih telah diseimbangkan (Balanced) agar model adil dalam memprediksi.")

    # --- 4. ARSITEKTUR MODEL ---
    elif pilihan == "4. Arsitektur Model":
        st.header("🧠 4. Arsitektur Model: Hybrid FastText + Bi-LSTM")
        
        st.markdown("""
        Kami menggunakan arsitektur **Hybrid** yang menggabungkan keunggulan representasi kata FastText dengan kemampuan memori jangka panjang Bi-LSTM.
        """)

        c_text, c_spacer, c_img = st.columns([1.5, 0.2, 1])
        
        with c_text:
            st.subheader("Rincian Layer & Fungsinya:")
            st.markdown("""
            1.  **Embedding (FastText):** Mengubah indeks kata menjadi vektor padat (300 dimensi). FastText unggul menangani kata *out-of-vocabulary* (typo/slang).
            2.  **SpatialDropout1D (0.3):** Mematikan sebagian neuron secara acak untuk mencegah model "menghafal" data (*overfitting*).
            3.  **Bi-LSTM (64 Units):** Memproses urutan kata dari dua arah (Depan-ke-Belakang DAN Belakang-ke-Depan) untuk menangkap konteks kalimat secara utuh.
            4.  **Dense Layer (Softmax):** Layer output dengan 3 neuron yang menghasilkan probabilitas untuk kelas **Negatif, Netral, dan Positif**.
            """)
            
            param_data = {
                "Nama Layer": ["Embedding", "SpatialDropout", "Bi-LSTM", "Dropout & L2", "Dense Output"],
                "Output Shape": ["(None, 60, 300)", "(None, 60, 300)", "(None, 128)", "(None, 128)", "(None, 3)"],
                "Jml Parameter": ["Trainable", "0", "186,880", "0", "387"]
            }
            st.dataframe(pd.DataFrame(param_data), use_container_width=True)

        with c_spacer:
            st.empty()

        with c_img:
            st.caption("Visualisasi Alur Data:")
            try:
                graph = graphviz.Digraph(node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': '#E8F0FE'})
                graph.attr(rankdir='TB') 
                
                graph.node('I', 'Input Teks\n(Integer Encoded)', fillcolor='#FFEBEE')
                graph.node('E', 'Embedding Layer\n(FastText Weights)', fillcolor='#FFF3E0') 
                graph.node('L', 'Bi-LSTM Layer\n(Proses 2 Arah)', fillcolor='#E3F2FD') 
                graph.node('D', 'Dense & Softmax\n(Klasifikasi)', fillcolor='#E8F5E9')
                
                graph.edge('I', 'E')
                graph.edge('E', 'L')
                graph.edge('L', 'D')
                
                st.graphviz_chart(graph, use_container_width=True)
            except:
                st.info("Install graphviz untuk melihat diagram alir.")

    # ==============================================================================
    # 5. EVALUASI MODEL 
    # ==============================================================================
    elif pilihan == "5. Evaluasi Model":
        st.header("5. Evaluasi Performa Model")
        st.markdown("Pengujian dilakukan menggunakan data yang **belum pernah dilihat** oleh model sebelumnya (Data Testing 20%).")
        
        tab_a, tab_b = st.tabs(["📊 Metrik Kuantitatif (Tabel)", "📉 Grafik Visualisasi (Interaktif)"])
        
        # --- TAB A: TABEL ANGKA ---
        with tab_a:
            st.subheader("1. Classification Report")
            st.markdown("""
            - **Precision:** Ketepatan prediksi (Minim *False Positive*).
            - **Recall:** Kelengkapan prediksi (Minim *False Negative*).
            - **F1-Score:** Rata-rata harmonis (Metrik utama untuk data tidak seimbang).
            """)
            
            path_perf = 'model/Tabel_Performa_LSTM.csv'
            if not os.path.exists(path_perf): path_perf = 'Tabel_Performa_LSTM.csv'

            if os.path.exists(path_perf):
                df_perf = pd.read_csv(path_perf, index_col=0)
                
                st.dataframe(
                    df_perf.style.highlight_max(axis=0, props='background-color: #FFEB3B; color: black; font-weight: bold'),
                    use_container_width=True
                )
                
                # Metric Cards
                if 'accuracy' in df_perf.index:
                    acc = df_perf.loc['accuracy', 'f1-score']
                    st.metric("Akurasi Total (Data Testing)", f"{acc*100:.2f}%")
            else:
                st.warning("⚠️ File 'Tabel_Performa_LSTM.csv' belum tersedia.")

        # --- TAB B: GRAFIK ---
        with tab_b:
            
            # 1. KURVA PEMBELAJARAN (Line Chart)
            st.subheader("1. Kurva Pembelajaran (Learning Curve)")
            st.info("ℹ️ Grafik ini dibuat dinamis dari file `Riwayat_Training.csv`. Anda bisa zoom in/out.")
            
            path_history = 'model/Riwayat_Training.csv'
            if not os.path.exists(path_history): path_history = 'Riwayat_Training.csv'

            if os.path.exists(path_history):
                df_hist = pd.read_csv(path_history)
                
                sub_tab1, sub_tab2 = st.tabs(["Akurasi", "Loss"])
                
                with sub_tab1:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=df_hist.index+1, y=df_hist['accuracy'], mode='lines+markers', name='Training Acc'))
                    fig_acc.add_trace(go.Scatter(x=df_hist.index+1, y=df_hist['val_accuracy'], mode='lines+markers', name='Validation Acc'))
                    fig_acc.update_layout(title="Pergerakan Akurasi per Epoch", xaxis_title="Epoch", yaxis_title="Akurasi", hovermode="x unified")
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with sub_tab2:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=df_hist.index+1, y=df_hist['loss'], mode='lines+markers', name='Training Loss', line=dict(color='orange')))
                    fig_loss.add_trace(go.Scatter(x=df_hist.index+1, y=df_hist['val_loss'], mode='lines+markers', name='Validation Loss', line=dict(color='red')))
                    fig_loss.update_layout(title="Pergerakan Loss per Epoch", xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified")
                    st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.warning("⚠️ File 'Riwayat_Training.csv' tidak ditemukan. Harap simpan history.history ke CSV saat di Colab.")

            st.markdown("---") 

            # 2. CONFUSION MATRIX (Heatmap)
            st.subheader("2. Confusion Matrix")
            
            path_cm = 'model/Data_Confusion_Matrix.csv' 
            if not os.path.exists(path_cm): path_cm = 'Data_Confusion_Matrix.csv'

            c_kiri, c_tengah, c_kanan = st.columns([1, 3, 1])
            with c_tengah:
                if os.path.exists(path_cm):
                    df_cm_data = pd.read_csv(path_cm)
                    
                    if 'y_true' in df_cm_data.columns and 'y_pred' in df_cm_data.columns:
                        labels = ['Negatif', 'Netral', 'Positif'] 
                        cm = confusion_matrix(df_cm_data['y_true'], df_cm_data['y_pred'])
                        
                        fig_cm = px.imshow(cm, 
                                           text_auto=True, 
                                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                                           x=labels, y=labels,
                                           color_continuous_scale='Blues')
                        fig_cm.update_layout(title="Matrix Kebenaran Prediksi")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        st.error("CSV Confusion Matrix harus punya kolom 'y_true' dan 'y_pred'.")
                else:
                    st.warning("⚠️ File 'Data_Confusion_Matrix.csv' tidak ditemukan.")

    # ==============================================================================
    # 6. TOPIC MODELING (LDA) 
    # ==============================================================================
    elif pilihan == "6. Topic Modeling (LDA)":
        st.header("6. Topic Modeling (LDA)")
        st.markdown("""
        **Tujuan:** Menggali "Apa yang sebenarnya dibicarakan?" di balik sentimen tersebut menggunakan algoritma **Latent Dirichlet Allocation (LDA)**.
        """)

        # --- BAGIAN A: METRIK EVALUASI (COHERENCE SCORE) ---
        st.subheader("A. Optimasi Jumlah Topik (Coherence Score)")
        st.info("💡 Grafik ini menunjukkan bagaimana kita menentukan jumlah topik terbaik. Titik tertinggi adalah jumlah topik yang paling optimal.")

        col_lda1, col_lda2 = st.columns([2, 1])
        
        with col_lda1:
            path_coherence = 'model/Nilai_Coherence.csv' 
            if not os.path.exists(path_coherence): path_coherence = 'Nilai_Coherence.csv'

            if os.path.exists(path_coherence):
                df_coh = pd.read_csv(path_coherence)
                
                # Plot Line Chart
                fig_coh = px.line(df_coh, x='Num_Topics', y='Coherence_Score', markers=True,
                                  title="Nilai Coherence Score vs Jumlah Topik",
                                  labels={'Num_Topics': 'Jumlah Topik', 'Coherence_Score': 'Skor Koherensi'})
                
                max_score = df_coh['Coherence_Score'].max()
                best_topic_num = df_coh.loc[df_coh['Coherence_Score'].idxmax(), 'Num_Topics']
                
                fig_coh.add_annotation(x=best_topic_num, y=max_score,
                                       text=f"Optimal: {int(best_topic_num)} Topik",
                                       showarrow=True, arrowhead=1)
                
                st.plotly_chart(fig_coh, use_container_width=True)
            else:
                st.warning("⚠️ File 'Nilai_Coherence.csv' tidak ditemukan.")
        
        with col_lda2:
            st.markdown("### 📝 Interpretasi:")
            st.write("""
            Model mencari pola kata yang sering muncul bersamaan. 
            
            **Coherence Score** mengukur seberapa "nyambung" kata-kata dalam satu topik. Skor yang tinggi menunjukkan topik tersebut mudah dipahami oleh manusia.
            """)

        st.markdown("---")

        # --- BAGIAN B: VISUALISASI TOPIK (BAR CHART DARI CSV) ---
        st.subheader("B. Visualisasi Kata Kunci per Topik")
        st.write("Berikut adalah kata-kata dominan yang membentuk setiap topik berdasarkan sentimen.")

        path_lda = 'model/Hasil_Analisis_Topik_LDA.csv'
        if not os.path.exists(path_lda): path_lda = 'Hasil_Analisis_Topik_LDA.csv'

        if os.path.exists(path_lda):
            try:
                df_lda = pd.read_csv(path_lda)
                
                # Fungsi Parsing: "0.035*kata" -> DataFrame
                def parse_lda_string(text_data):
                    data_items = []
                    for item in str(text_data).split(', '):
                        try:
                            weight, word = item.split('*')
                            data_items.append({'Kata': word.strip(), 'Bobot': float(weight)})
                        except: continue
                    return pd.DataFrame(data_items).sort_values(by='Bobot', ascending=True)

                # Tabs untuk Topik
                t_neg, t_net, t_pos = st.tabs(["Topik Negatif", "Topik Netral", "Topik Positif"])
                mapping = {'negatif': t_neg, 'netral': t_net, 'positif': t_pos}

                for sentimen, tab in mapping.items():
                    with tab:
                        # Filter CSV berdasarkan sentimen
                        df_subset = df_lda[df_lda['Sentimen'] == sentimen]
                        
                        if df_subset.empty:
                            st.warning(f"Belum ada data topik untuk {sentimen}.")
                        else:
                            # Tampilkan Topik
                            for idx, row in df_subset.iterrows():
                                topik_ke = row['Topik Ke']
                                df_chart = parse_lda_string(row['Kata Kunci (Bobot)'])
                                
                                if not df_chart.empty:
                                    # Plot Bar Chart Horizontal
                                    fig = px.bar(
                                        df_chart, x='Bobot', y='Kata', orientation='h',
                                        title=f"<b>Topik {topik_ke}:</b> Kata Kunci Dominan",
                                        color='Bobot',
                                        color_continuous_scale='Blues' if sentimen == 'negatif' else 'Greys' if sentimen == 'netral' else 'Greens'
                                    )
                                    fig.update_layout(height=300, showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.divider()
            except Exception as e:
                st.error(f"Gagal memproses data LDA: {e}")
        else:
            st.warning("⚠️ File 'Hasil_Analisis_Topik_LDA.csv' belum tersedia.")