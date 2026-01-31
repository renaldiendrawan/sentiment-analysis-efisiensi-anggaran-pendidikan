import streamlit as st
from streamlit_option_menu import option_menu

# --- IMPORT MODUL LOKAL ---
from utils import load_resources
from views.beranda import render_beranda    
from views.visualisasi import render_visualisasi
from views.proses_data import render_proses_data
from views.analisis_teks import render_analisis_teks
from views.analisis_csv import render_analisis_csv

# ==============================================================================
# 1. SETUP KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Isu Efisiensi Anggaran Sektor Pendidikan",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model Bi-LSTM & FastText Tokenizer 
model, tokenizer = load_resources()

# ==============================================================================
# 2. SIDEBAR NAVIGATION (MENU KIRI)
# ==============================================================================
with st.sidebar:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image("images/data_analytics.png", use_column_width=True)

    st.markdown(
        """
        <h2 style='text-align: center; margin-top: 10px; margin-bottom: 5px; font-weight: 800; font-size: 26px; line-height: 1.2;'>
            Sistem Analisis Sentimen
        </h2>
        <p style='text-align: center; color: gray; font-size: 14px;'>
            Kebijakan Efisiensi Anggaran Pendidikan
        </p>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Beranda", "Visualisasi", "Proses Data", "Analisis Teks", "Analisis File CSV"],
        icons=["house", "bar-chart", "gear", "chat-text", "file-earmark-spreadsheet"],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#007BFF"} # Warna biru aktif
        }
    )
    
    st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True) 

    bot_col1, bot_col2, bot_col3 = st.columns([1, 5, 1])
    with bot_col2:
        st.image("images/logo_jti.png", use_column_width=True)

    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 13px; margin-top: 25px; margin-bottom: 40px;'>
            © 2026 - Skripsi<br>
            <b>Renaldi Endrawan</b>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ==============================================================================
# 3. ROUTING HALAMAN (MENAMPILKAN KONTEN)
# ==============================================================================
if selected == "Beranda":
    render_beranda()
elif selected == "Visualisasi":
    render_visualisasi()
elif selected == "Proses Data":
    render_proses_data()
elif selected == "Analisis Teks":
    render_analisis_teks(model, tokenizer) 
elif selected == "Analisis File CSV":
    render_analisis_csv(model, tokenizer)