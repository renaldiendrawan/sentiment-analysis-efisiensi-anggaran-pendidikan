import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

from utils import predict_sentiment 

HISTORY_FILE = 'data/riwayat_analisis.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except: return []
    return []

def save_history(data):
    os.makedirs('data', exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# ==============================================================================
# Fungsi Clear sekarang menghapus Teks DAN Hasil Prediksi
# ==============================================================================
def clear_input():
    st.session_state['input_teks_analisis'] = ""
    st.session_state['latest_result'] = None 

# ==============================================================================
# RENDER HALAMAN UTAMA
# ==============================================================================
def render_analisis_teks(model, tokenizer):
    st.title("💬 Analisis Sentimen (Single Text)")
    st.markdown("Ketikkan kalimat opini terkait kebijakan efisiensi anggaran pendidikan, dan biarkan AI memprediksi sentimennya secara *real-time*.")

    # 1. INISIALISASI SESSION STATE
    if 'history_analisis' not in st.session_state:
        st.session_state['history_analisis'] = load_history()
    if 'latest_result' not in st.session_state:
        st.session_state['latest_result'] = None
    if 'show_confirm' not in st.session_state:
        st.session_state['show_confirm'] = False
    if 'rows_to_delete' not in st.session_state:
        st.session_state['rows_to_delete'] = []
    if 'input_teks_analisis' not in st.session_state:
        st.session_state['input_teks_analisis'] = ""

    # ==============================================================================
    # 2. AREA INPUT TEKS & TOMBOL
    # ==============================================================================
    input_text = st.text_area(
        "Masukkan Teks Opini di sini:", 
        height=150, 
        placeholder="Contoh: Sangat kecewa anggaran KIP Kuliah dipotong...",
        key='input_teks_analisis'
    )

    # ==============================================================================
    # Deteksi Hapus Manual (Backspace)
    # ==============================================================================
    if not input_text.strip():
        st.session_state['latest_result'] = None
    
    col_btn1, col_btn2, col_spacer = st.columns([2, 2, 6])
    
    with col_btn1:
        btn_analisis = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
    
    with col_btn2:
        st.button("🧹 Bersihkan Teks", on_click=clear_input, use_container_width=True)

    if btn_analisis:
        if input_text.strip():
            with st.spinner('🤖 Model Bi-LSTM sedang memproses teks...'):
                label, confidence, probs, clean_txt = predict_sentiment(input_text, model, tokenizer)
                
                probabilitas_bersih = [float(p) for p in probs]

                st.session_state['latest_result'] = {
                    "label": label,
                    "confidence": confidence,
                    "probs": probabilitas_bersih,
                    "clean_txt": clean_txt
                }

                waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_entry = {
                    "Waktu": waktu_sekarang,
                    "Teks Asli": input_text,
                    "Teks Bersih": clean_txt,
                    "Label": label,
                    "Keyakinan (%)": round(confidence, 2)
                }
                st.session_state['history_analisis'].append(new_entry)
                save_history(st.session_state['history_analisis'])
        else:
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu.")

    # ==============================================================================
    # 3. AREA HASIL PREDIKSI
    # ==============================================================================
    if st.session_state['latest_result']:
        res = st.session_state['latest_result']
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.subheader("🎯 Hasil Prediksi")
            if res['label'] == "Positif": st.success(f"**🟢 SENTIMEN POSITIF**")
            elif res['label'] == "Negatif": st.error(f"**🔴 SENTIMEN NEGATIF**")
            else: st.warning(f"**⚪ SENTIMEN NETRAL**")
            st.metric("Tingkat Keyakinan (Confidence)", f"{res['confidence']:.2f}%")
        
        with col_res2:
            st.subheader("📊 Distribusi Probabilitas")
            st.caption("Detail perhitungan matematis model (Total 100%)")
            
            st.write(f"🔴 **Negatif:** {res['probs'][0]*100:.1f}%")
            st.progress(res['probs'][0])

            st.write(f"⚪ **Netral:** {res['probs'][1]*100:.1f}%")
            st.progress(res['probs'][1])

            st.write(f"🟢 **Positif:** {res['probs'][2]*100:.1f}%")
            st.progress(res['probs'][2])
        
        st.markdown("#### 🔍 Teks Hasil Preprocessing (Cleaning & Normalisasi)")
        st.info(f"{res['clean_txt']}")

    st.markdown("---")

    # ==============================================================================
    # 4. AREA HISTORY 
    # ==============================================================================
    st.subheader("📚 Riwayat Analisis")

    if len(st.session_state['history_analisis']) > 0:
        # 1. Siapkan Data
        df_history = pd.DataFrame(st.session_state['history_analisis'])
        df_display = df_history.iloc[::-1].reset_index(drop=True) 
        
        if 'Pilih' not in df_display.columns:
            df_display.insert(0, "Pilih", False)

        # 2. Filter & Select All
        c_search, c_all = st.columns([3, 1])
        with c_search:
            q = st.text_input("Cari:", placeholder="Filter riwayat...", label_visibility="collapsed")
        with c_all:
            if st.checkbox("Pilih Semua"):
                df_display['Pilih'] = True

        if q:
            df_display = df_display[df_display['Teks Asli'].str.contains(q, case=False, na=False)]

        # 3. Tabel Editor
        with st.container():
            edited_df = st.data_editor(
                df_display,
                column_config={
                    "Pilih": st.column_config.CheckboxColumn("Hapus?", width="small", default=False),
                    "Waktu": st.column_config.TextColumn("Waktu", disabled=True),
                    "Teks Asli": st.column_config.TextColumn("Teks Tweet", disabled=True),
                    "Label": st.column_config.TextColumn("Prediksi", disabled=True),
                    "Keyakinan (%)": st.column_config.NumberColumn("Score", format="%.2f%%", disabled=True)
                },
                hide_index=True,
                use_container_width=True,
                key="history_editor"
            )

        # 4. Tombol Aksi (LAYOUT BARU)
        selected_rows = edited_df[edited_df['Pilih'] == True]
        count = len(selected_rows)
        
        popup_placeholder = st.empty() 

        # --- BARIS 1: TOMBOL HAPUS (Merah & Primary) ---
        col_del_1, col_del_2 = st.columns(2)

        with col_del_1:
            if st.button(f"🗑️ Hapus ({count}) Item", type="primary", disabled=count==0, use_container_width=True):
                st.session_state['rows_to_delete'] = selected_rows['Waktu'].tolist()
                st.session_state['show_confirm'] = True

        with col_del_2:
            if st.button("🚨 Hapus Semua", type="secondary", use_container_width=True):
                st.session_state['rows_to_delete'] = "ALL"
                st.session_state['show_confirm'] = True

        # --- BARIS 2: TOMBOL DOWNLOAD (Hijau/Standar - Di Bawah) ---
        st.write("") 
        
        csv_data = df_display.drop(columns=['Pilih']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV (Backup Data Riwayat)", 
            data=csv_data, 
            file_name="Riwayat_Analisis.csv", 
            mime="text/csv", 
            use_container_width=True
        )

        # --- 5. LOGIKA POP-UP KONFIRMASI ---
        if st.session_state.get('show_confirm', False):
            with popup_placeholder.container():
                st.markdown("---")
                msg = "SEMUA DATA" if st.session_state['rows_to_delete'] == "ALL" else f"{len(st.session_state['rows_to_delete'])} DATA TERPILIH"
                
                with st.chat_message("assistant", avatar="⚠️"):
                    st.write(f"**KONFIRMASI:** Apakah Anda yakin ingin menghapus **{msg}**?")
                    st.caption("Tindakan ini tidak dapat dibatalkan.")
                    
                    col_yes, col_no = st.columns([1, 4])
                    with col_yes:
                        if st.button("✅ YA, Hapus"):
                            if st.session_state['rows_to_delete'] == "ALL":
                                st.session_state['history_analisis'] = []
                            else:
                                targets = st.session_state['rows_to_delete']
                                st.session_state['history_analisis'] = [
                                    item for item in st.session_state['history_analisis'] 
                                    if item['Waktu'] not in targets
                                ]
                            
                            save_history(st.session_state['history_analisis'])
                            st.session_state['show_confirm'] = False
                            st.session_state['rows_to_delete'] = []
                            st.success("Berhasil dihapus!")
                            st.rerun() 
                    
                    with col_no:
                        if st.button("❌ Batal"):
                            st.session_state['show_confirm'] = False
                            st.session_state['rows_to_delete'] = []
                            st.rerun()
                st.markdown("---")

    else:
        st.info("📝 Belum ada riwayat analisis.")