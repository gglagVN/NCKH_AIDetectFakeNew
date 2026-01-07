import os
import streamlit as st
import torch
import joblib
import re
import unicodedata
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel

# LOAD ASSETS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Code", "models", "svm_model.pkl")

@st.cache_resource
def load_all():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    classifier = joblib.load(MODEL_PATH)
    return tokenizer, phobert, classifier

tokenizer, phobert, classifier = load_all()

def clean_standard(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_sw(text):
    sw = ["là", "của", "những", "cái", "rằng", "thì", "mà", "vậy", "với", "cho"]
    return " ".join([w for w in text.split() if w not in sw])

st.title("🛡️ VN Fake News Detector")
input_txt = st.text_area("Dán nội dung cần kiểm tra:", height=250)

if st.button("🔍 PHÂN TÍCH"):
    if not input_txt.strip():
        st.warning("Vui lòng nhập văn bản.")
    else:
        with st.spinner("Đang xử lý..."):
            # Tiền xử lý
            t = clean_standard(input_txt)
            t = ViTokenizer.tokenize(t)
            t = remove_sw(t)
            
            # Embedding
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
            with torch.no_grad():
                emb = phobert(**inputs).last_hidden_state[:, 0, :].numpy()
            
            # Dự đoán
            probs = classifier.predict_proba(emb)[0]
            if probs[1] > probs[0]:
                st.error(f"🚨 KẾT QUẢ: TIN GIẢ ({probs[1]*100:.2f}%)")
            else:
                st.success(f"✅ KẾT QUẢ: TIN THẬT ({probs[0]*100:.2f}%)")
            
            st.bar_chart({"Tin thật": probs[0], "Tin giả": probs[1]})