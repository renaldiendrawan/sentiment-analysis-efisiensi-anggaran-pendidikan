import pandas as pd
import numpy as np
import json
import h5py
import re
import pickle
import os
import streamlit as st
import tensorflow as tf

try:
    from tensorflow.keras.utils import pad_sequences
except ImportError:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# 1. KONFIGURASI GLOBAL
# ==============================================================================
MAX_SEQUENCE_LENGTH = 100
MODEL_PATH = 'model/Model_Sentiment_BiLSTM.h5'
TOKENIZER_JSON_PATH = 'model/tokenizer_sentiment.json'
TOKENIZER_PICKLE_PATH = 'model/tokenizer_sentiment.pickle'

# ==============================================================================
# 2. PATCHING MODEL
# ==============================================================================
def recursive_fix_config(config):
    """Memperbaiki konfigurasi model agar bisa dibaca di berbagai versi TF"""
    if isinstance(config, list):
        return [recursive_fix_config(x) for x in config]
    if isinstance(config, dict):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        if 'dtype' in config:
            if isinstance(config['dtype'], dict) or 'Policy' in str(config['dtype']):
                config['dtype'] = 'float32'
        for key, value in config.items():
            config[key] = recursive_fix_config(value)
    return config

# ==============================================================================
# 3. LOAD RESOURCES (MODEL & TOKENIZER)
# ==============================================================================
@st.cache_resource
def load_resources():
    model = None
    tokenizer = None

    # --- A. LOAD MODEL ---
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
        return None, None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        try:
            with h5py.File(MODEL_PATH, mode='r') as f:
                model_config_str = f.attrs.get('model_config')
                if isinstance(model_config_str, bytes):
                    model_config_str = model_config_str.decode('utf-8')
                
                model_config_dict = json.loads(model_config_str)
                fixed_config = recursive_fix_config(model_config_dict)
                
                model = tf.keras.models.model_from_json(json.dumps(fixed_config))
                model.load_weights(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return None, None

    # --- B. LOAD TOKENIZER ---
    try:
        if os.path.exists(TOKENIZER_JSON_PATH):
            with open(TOKENIZER_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                
                try:
                    parsed_json = json.loads(content)
                    
                    if isinstance(parsed_json, str):
                        input_tokenizer = parsed_json
                    
                    else:
                        input_tokenizer = json.dumps(parsed_json)
                        
                except:
                    input_tokenizer = content
                
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(input_tokenizer)

        elif os.path.exists(TOKENIZER_PICKLE_PATH):
            with open(TOKENIZER_PICKLE_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            st.error("❌ File Tokenizer tidak ditemukan.")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Gagal memuat tokenizer: {e}")
        return None, None
        
    return model, tokenizer

# ==============================================================================
# 4. PREPROCESSING TEKS
# ==============================================================================
slang_dict = {
    "yg": "yang", "dgn": "dengan", "gak": "tidak", "ga": "tidak", 
    "tp": "tapi", "bgt": "banget", "udah": "sudah", "aja": "saja",
    "jd": "jadi", "d": "di", "sprt": "seperti", "opr": "operasional",
    "sdh": "sudah", "tlg": "tolong", "krn": "karena", "jgn": "jangan",
    "tdk": "tidak", "kalo": "kalau", "klo": "kalau", "blm": "belum",
    "bkn": "bukan", "tak": "tidak", "tau": "tahu", "aq": "aku", 
    "km": "kamu", "bs": "bisa", "dlm": "dalam", "utk": "untuk"
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

# ==============================================================================
# 5. PREDIKSI
# ==============================================================================
def predict_sentiment(text, model, tokenizer):
    if not text or not model or not tokenizer:
        return "Error", 0.0, [0, 0, 0], text

    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    prediction = model.predict(padded)[0]
    
    labels = ['Negatif', 'Netral', 'Positif'] 
    label_idx = np.argmax(prediction)
    label = labels[label_idx]
    confidence = prediction[label_idx] * 100
    
    return label, confidence, prediction, cleaned_text