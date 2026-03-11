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
        * **Alasan:** Model Deep Learning (seperti LSTM) membutuhkan konteks kalimat utuh untuk memahami nuansa sentimen (contoh: kata *"tidak"* sangat penting untuk membalikkan makna *"suka"* menjadi *"tidak suka"*). Menghapus *stopword* justru dapat merusak tata bahasa yang akan dibaca oleh model secara sekuensial.
        """)

        with st.expander("ℹ️ Rincian 5 Langkah Preprocessing", expanded=True):
            st.markdown("""
            1.  **Case Folding:** Menyeragamkan huruf menjadi kecil (*lowercase*).
            2.  **Cleaning:** Menghapus elemen non-teks (URL, Mention `@`, Hashtag `#`, Angka, Tanda Baca).
            3.  **Tokenizing:** Memecah kalimat menjadi potongan kata per kata.
            4.  **Normalisasi Slang:** Mengubah kata tidak baku (*bgt, gk, sy*) menjadi baku (*banget, tidak, saya*) menggunakan kamus *lexicon*.
            5.  **Detokenizing:** Menggabungkan kata kembali menjadi kalimat utuh.
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
        st.write("Setiap kata unik dalam dataset diberi ID angka. Karena panjang tweet berbeda-beda, kita lakukan **Padding (Post)** agar semua input memiliki panjang seragam (**100 kata**). Angka 0 di akhir akan diabaikan oleh fitur *Masking* pada model.")

        if not df_mentah.empty and 'Label' in df_mentah.columns:
            df_token = df_mentah.dropna(subset=['Label']).copy()
            
            # Helper simulasi token
            def get_word_id(word): return int(hashlib.md5(word.encode()).hexdigest(), 16) % 3000 + 1
            
            df_token['Detail Token'] = df_token['Tweet_Final'].apply(lambda t: ", ".join([f"{w}:{get_word_id(w)}" for w in str(t).split()[:10]]))
            df_token['Padding Sequence (100)'] = df_token['Tweet_Final'].apply(lambda t: str(([get_word_id(w) for w in str(t).split()] + [0]*100)[:20]) + " ...")
            
            st.dataframe(df_token[['Tweet_Final', 'Detail Token', 'Padding Sequence (100)']], use_container_width=True)

            st.markdown("---")
            st.subheader("B. Splitting 80:20 & Skenario 5 Percobaan")
            
            st.markdown("""
            **Skenario Pelatihan:**
            Model dilatih menggunakan **5 Skenario Percobaan** (P1 hingga P5) dengan porsi data latih masing-masing 20%, 40%, 60%, 80%, dan 100% (dari total 80% split data latih).
            
            **Penanganan Imbalance (ROS):**
            Kami menduplikasi data minoritas (Positif/Netral) secara acak (*Random Over Sampling*) di **setiap porsi data latih** hingga jumlahnya setara dengan kelas mayoritas (Negatif). Data Testing (20%) dibiarkan murni agar evaluasi tetap objektif.
            """)

            df_train, df_test = train_test_split(df_token, test_size=0.2, random_state=42, stratify=df_token['Label'])
            kelas_mayoritas = df_train['Label'].value_counts().max()
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Maksimal Data Latih (80%)", f"{len(df_train):,} Sample", "Skenario P5")
            col_metric2.metric("Data Uji Tetap (20%)", f"{len(df_test):,} Sample", "Validasi Objektif")
            col_metric3.metric("Target ROS P5", f"{kelas_mayoritas}", "Per Kelas Sentimen")
            
            st.success(f"✅ **Status Data:** Dataset latih telah diseimbangkan (Balanced) menggunakan teknik ROS pada tahapan pemodelan.")

    # --- 4. ARSITEKTUR MODEL ---
    elif pilihan == "4. Arsitektur Model":
        st.header("🧠 4. Arsitektur Model: LSTM Standar")
        
        st.markdown("""
        Kami menggunakan arsitektur **Long Short-Term Memory (LSTM)** yang dipadukan dengan *Keras Embedding Layer* dan fitur *Masking*.
        """)

        c_text, c_spacer, c_img = st.columns([1.5, 0.2, 1])
        
        with c_text:
            st.subheader("Rincian Layer & Fungsinya:")
            st.markdown("""
            1.  **Embedding (Keras):** Mengubah indeks kata menjadi vektor padat (128 dimensi). Fitur `mask_zero=True` diaktifkan agar model murni fokus pada teks tanpa terdistraksi oleh angka padding (0) di akhir kalimat.
            2.  **SpatialDropout1D (0.2):** Mematikan sebagian 1D feature maps secara acak untuk mencegah model "menghafal" data secara berlebihan (*overfitting*).
            3.  **LSTM (64 Units):** Memproses urutan kata secara sekuensial (dari awal hingga akhir kalimat) agar model bisa memahami relasi dan pola frasa sentimen dengan sangat baik.
            4.  **Dense Layer (32 Units):** Ekstraksi fitur tingkat tinggi menggunakan fungsi aktivasi ReLU dengan peluruhan (Dropout 0.2).
            5.  **Dense Output (3 Units):** Layer akhir dengan aktivasi *Softmax* yang menghasilkan nilai probabilitas klasifikasi untuk **Negatif, Netral, dan Positif**.
            """)
            
            param_data = {
                "Nama Layer": ["Embedding", "SpatialDropout", "LSTM", "Dense", "Dense Output"],
                "Output Shape": ["(None, 100, 128)", "(None, 100, 128)", "(None, 64)", "(None, 32)", "(None, 3)"],
                "Jml Parameter": ["1,280,000", "0", "49,408", "2,080", "99"]
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
                graph.node('E', 'Embedding Layer\n(Dimensi 128, Masking)', fillcolor='#FFF3E0') 
                graph.node('L', 'LSTM Layer\n(Proses Sekuensial)', fillcolor='#E3F2FD') 
                graph.node('D', 'Dense & Softmax\n(Klasifikasi 3 Kelas)', fillcolor='#E8F5E9')
                
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
        st.header("5. Evaluasi Performa Model (Skenario P1-P5)")
        st.markdown("Evaluasi ini mencakup perbandingan 5 skenario pelatihan berdasarkan ukuran rasio data latih (20% hingga 100%), yang diuji menggunakan **Data Testing murni (20%)**.")
        
        tab_a, tab_b, tab_c = st.tabs(["📊 Metrik (Model P5)", "📈 Perbandingan 5 Skenario", "📉 Detail Learning Curve"])
        
        # --- TAB A: TABEL ANGKA ---
        with tab_a:
            st.subheader("1. Classification Report (Model P5)")
            st.markdown("""
            - **Precision:** Ketepatan prediksi model (Meminimalisir salah tebak positif palsu).
            - **Recall:** Kelengkapan prediksi (Meminimalisir salah tebak negatif palsu).
            - **F1-Score:** Rata-rata harmonis antara Precision dan Recall.
            """)
            
            path_perf = 'model/Tabel_Performa_LSTM.csv'
            if not os.path.exists(path_perf): path_perf = 'Tabel_Performa_LSTM.csv'

            if os.path.exists(path_perf):
                df_perf = pd.read_csv(path_perf, index_col=0)
                st.table(
                    df_perf.style.highlight_max(axis=0, props='background-color: #FFEB3B; color: black; font-weight: bold')
                )
                if 'accuracy' in df_perf.index:
                    acc = df_perf.loc['accuracy', 'f1-score']
                    st.metric("Akurasi Total (Data Testing P5)", f"{acc*100:.2f}%")
            else:
                st.warning("⚠️ File 'Tabel_Performa_LSTM.csv' belum tersedia.")
            
            st.markdown("---")
            st.subheader("2. Confusion Matrix (Model P5)")
            path_cm = 'model/Data_Confusion_Matrix.csv' 
            if os.path.exists(path_cm):
                df_cm_data = pd.read_csv(path_cm)
                if 'y_true' in df_cm_data.columns and 'y_pred' in df_cm_data.columns:
                    labels = ['Negatif', 'Netral', 'Positif'] 
                    cm = confusion_matrix(df_cm_data['y_true'], df_cm_data['y_pred'])
                    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prediksi Model", y="Label Aktual (Asli)", color="Jumlah Data"), x=labels, y=labels, color_continuous_scale='Blues')
                    fig_cm.update_layout(title="Matrix Kebenaran Prediksi P5")
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning("⚠️ File 'Data_Confusion_Matrix.csv' tidak ditemukan.")

        # --- TAB B: BAR CHART PERBANDINGAN SKENARIO (DINAMIS DARI CSV) ---
        with tab_b:
            st.subheader("Perbandingan Akurasi Skenario P1 hingga P5")
            st.markdown("Grafik interaktif ini menunjukkan bahwa semakin besar porsi data latih yang diberikan, maka kemampuan model dalam mengklasifikasi sentimen cenderung semakin baik.")
            
            path_akurasi = 'model/Akurasi_Skenario.csv'
            if os.path.exists(path_akurasi):
                df_acc_skenario = pd.read_csv(path_akurasi)
                rata_rata = df_acc_skenario['Akurasi'].mean()
                
                # Buat label gabungan P1 (20%), dst
                df_acc_skenario['Label_X'] = df_acc_skenario['Skenario'] + " (" + df_acc_skenario['Porsi_Data'] + ")"
                
                fig_bar = px.bar(
                    df_acc_skenario, x='Label_X', y='Akurasi', 
                    text='Akurasi', 
                    color='Skenario',
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    title="Persentase Akurasi per Skenario Data Latih",
                    labels={'Label_X': 'Skenario (Porsi Data Latih)', 'Akurasi': 'Akurasi (%)'}
                )
                
                fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_bar.add_hline(y=rata_rata, line_dash="dot", line_color="red", annotation_text=f"Rata-rata: {rata_rata:.2f}%")
                fig_bar.update_layout(yaxis_range=[0, 100], showlegend=False)
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("⚠️ File 'Akurasi_Skenario.csv' belum tersedia. Harap export dari Colab.")

        # --- TAB C: KURVA PEMBELAJARAN SEMUA SKENARIO (DINAMIS DARI CSV) ---
        with tab_c:
            st.subheader("Grafik Pergerakan Learning Curve")
            st.info("Pilih skenario di bawah ini untuk melihat detail pergerakan Akurasi dan Loss-nya secara interaktif.")
            
            path_hist_semua = 'model/Riwayat_Training_Semua.csv'
            if os.path.exists(path_hist_semua):
                df_all_hist = pd.read_csv(path_hist_semua)
                
                # Opsi interaktif untuk memilih Skenario
                skenario_pilihan = st.selectbox("Pilih Skenario:", ['P1', 'P2', 'P3', 'P4', 'P5'], index=4)
                
                # Filter data berdasarkan skenario yang dipilih
                df_hist_filter = df_all_hist[df_all_hist['Skenario'] == skenario_pilihan]
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_acc_line = go.Figure()
                    fig_acc_line.add_trace(go.Scatter(x=df_hist_filter['Epoch'], y=df_hist_filter['accuracy'], mode='lines+markers', name='Train Acc'))
                    fig_acc_line.add_trace(go.Scatter(x=df_hist_filter['Epoch'], y=df_hist_filter['val_accuracy'], mode='lines+markers', name='Val Acc'))
                    fig_acc_line.update_layout(title=f"Akurasi ({skenario_pilihan})", xaxis_title="Epochs", yaxis_title="Akurasi", hovermode="x unified")
                    st.plotly_chart(fig_acc_line, use_container_width=True)
                
                with col_chart2:
                    fig_loss_line = go.Figure()
                    fig_loss_line.add_trace(go.Scatter(x=df_hist_filter['Epoch'], y=df_hist_filter['loss'], mode='lines+markers', name='Train Loss', line=dict(color='orange')))
                    fig_loss_line.add_trace(go.Scatter(x=df_hist_filter['Epoch'], y=df_hist_filter['val_loss'], mode='lines+markers', name='Val Loss', line=dict(color='red')))
                    fig_loss_line.update_layout(title=f"Loss ({skenario_pilihan})", xaxis_title="Epochs", yaxis_title="Loss", hovermode="x unified")
                    st.plotly_chart(fig_loss_line, use_container_width=True)
            else:
                st.warning("⚠️ File 'Riwayat_Training_Semua.csv' belum tersedia. Harap export dari Colab.")

    # ==============================================================================
    # 6. TOPIC MODELING (LDA) 
    # ==============================================================================
    elif pilihan == "6. Topic Modeling (LDA)":
        st.header("6. Topic Modeling (LDA)")
        st.markdown("""
        **Tujuan:** Menggali "Apa yang sebenarnya dibicarakan publik?" di balik masing-masing sentimen menggunakan metode **Latent Dirichlet Allocation (LDA)**.
        """)

        # --- BAGIAN A: METRIK EVALUASI (COHERENCE SCORE) ---
        st.subheader("A. Optimasi Jumlah Topik (Coherence Score)")
        st.info("💡 Grafik ini menunjukkan bagaimana model menentukan jumlah topik (K) terbaik secara ilmiah berdasarkan skor *Coherence c_v* tertinggi.")

        col_lda1, col_lda2 = st.columns([2, 1])
        
        with col_lda1:
            path_coherence = 'model/Nilai_Coherence.csv' 
            if not os.path.exists(path_coherence): path_coherence = 'Nilai_Coherence.csv'

            if os.path.exists(path_coherence):
                df_coh = pd.read_csv(path_coherence)
                
                # Plot Line Chart
                fig_coh = px.line(df_coh, x='Num_Topics', y='Coherence_Score', markers=True,
                                  title="Pergerakan Nilai Coherence Score",
                                  labels={'Num_Topics': 'Jumlah Topik', 'Coherence_Score': 'Skor Koherensi (c_v)'})
                
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
            Algoritma mesin bekerja dengan mencari pola kata yang sering muncul bersamaan di dalam satu dokumen teks. 
            
            **Coherence Score** bertugas untuk mengukur seberapa masuk akal ("nyambung") kumpulan kata-kata dalam satu topik. Semakin tinggi skornya, maka topik tersebut akan semakin mudah diinterpretasikan oleh pembaca/manusia.
            """)

        st.markdown("---")

        # --- BAGIAN B: VISUALISASI TOPIK (BAR CHART DARI CSV) ---
        st.subheader("B. Visualisasi Kata Kunci per Topik")
        st.write("Berikut adalah distribusi kata-kata kunci dominan yang mewakili setiap topik berdasarkan prediksi sentimen data *testing*.")

        path_lda = 'model/Hasil_Analisis_Topik_LDA.csv'
        if not os.path.exists(path_lda): path_lda = 'Hasil_Analisis_Topik_LDA.csv'

        if os.path.exists(path_lda):
            try:
                df_lda = pd.read_csv(path_lda)
                
                # Fungsi Parsing Teks dari format CSV
                def parse_lda_string(text_data):
                    data_items = []
                    # Memisahkan format yang sudah kita bersihkan di Colab
                    for word in str(text_data).split(','):
                        word = word.strip()
                        if word:
                            # Bobot diset dinamis untuk memunculkan visual Bar Horizontal (berdasarkan urutan)
                            data_items.append({'Kata': word})
                    
                    df_res = pd.DataFrame(data_items)
                    if not df_res.empty:
                        # Memberikan bobot buatan berdasarkan urutan (agar chart terbentuk rapi dari atas ke bawah)
                        df_res['Bobot'] = range(len(df_res), 0, -1)
                        df_res = df_res.sort_values(by='Bobot', ascending=True)
                    return df_res

                # Tabs untuk Topik
                t_neg, t_net, t_pos = st.tabs(["🔴 Topik Negatif", "⚪ Topik Netral", "🟢 Topik Positif"])
                mapping = {'negatif': t_neg, 'netral': t_net, 'positif': t_pos}

                for sentimen, tab in mapping.items():
                    with tab:
                        # Filter CSV berdasarkan sentimen
                        df_subset = df_lda[df_lda['Sentimen'].str.lower() == sentimen]
                        
                        if df_subset.empty:
                            st.warning(f"Belum ada data ekstraksi topik untuk sentimen {sentimen.upper()}.")
                        else:
                            col_t1, col_t2 = st.columns(2)
                            
                            # Tampilkan Topik dengan 2 kolom berjajar
                            for idx, row in df_subset.iterrows():
                                topik_ke = row['Topik Ke']
                                df_chart = parse_lda_string(row['Kata Kunci'])
                                
                                if not df_chart.empty:
                                    fig = px.bar(
                                        df_chart, x='Bobot', y='Kata', orientation='h',
                                        title=f"<b>Topik {topik_ke}</b>",
                                        color='Bobot',
                                        color_continuous_scale='Reds' if sentimen == 'negatif' else 'Greys' if sentimen == 'netral' else 'Greens'
                                    )
                                    # Sembunyikan X-axis karena ini hanya bobot representasi urutan
                                    fig.update_layout(height=280, showlegend=False, xaxis_title=None, xaxis_visible=False)
                                    
                                    if idx % 2 == 0:
                                        with col_t1: st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        with col_t2: st.plotly_chart(fig, use_container_width=True)
                                        
            except Exception as e:
                st.error(f"Gagal memproses visualisasi data LDA: {e}")
        else:
            st.warning("⚠️ File 'Hasil_Analisis_Topik_LDA.csv' belum tersedia di dalam folder model.")