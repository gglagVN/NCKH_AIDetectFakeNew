import os
import streamlit as st
import torch
import numpy as np
import joblib
import re
import unicodedata
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel

# ================= CẤU HÌNH & TẢI MÔ HÌNH =================
# Đảm bảo đường dẫn này khớp với máy của bạn
os.environ["HF_HOME"] = "G:/hf_cache"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Trỏ thẳng vào thư mục models/svm_model.pkl
MODEL_PATH = os.path.join(BASE_DIR, "Code", "models", "svm_model.pkl")

@st.cache_resource 
def load_assets():
    # Tải Tokenizer và PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    phobert.eval()
    
    # Tải classifier SVM (đã đạt 96.77%)
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
    classifier = joblib.load(MODEL_PATH)
    return tokenizer, phobert, classifier

tokenizer, phobert, classifier = load_assets()

# ================= TIỀN XỬ LÝ (PHẢI GIỐNG FILE TRAIN) =================
def clean_text_app(text):
    if not isinstance(text, str): return ""
    # 1. Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)
    # 2. Chuyển về chữ thường
    text = text.lower()
    # 3. Xóa URL, Email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    # 4. Xóa ký tự đặc biệt (GIỮ LẠI SỐ - Vì lúc train chúng ta giữ số)
    text = re.sub(r'[^\w\s]', ' ', text)
    # 5. Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_app(text):
    # Danh sách stopwords khớp với file prepare_data.py
    stopwords = ["là", "của", "những", "cái", "rằng", "thì", "mà", "vậy", "với", "cho"]
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

# ================= HÀM DỰ ĐOÁN =================
def predict_news(text):
    # Bước 1: Làm sạch
    text = clean_text_app(text)
    # Bước 2: Tách từ tiếng Việt (PyVi)
    text = ViTokenizer.tokenize(text)
    # Bước 3: Xóa stopword
    text = remove_stopwords_app(text)
    
    # Bước 4: Tokenize cho PhoBERT
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256, 
        padding="max_length"
    )
    
    # Bước 5: Lấy Embedding từ PhoBERT
    with torch.no_grad():
        outputs = phobert(**inputs)
        # Lấy vector [CLS] ở vị trí đầu tiên
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # Bước 6: Dự đoán bằng SVM
    probs = classifier.predict_proba(embedding)[0]
    return probs

# ================= GIAO DIỆN STREAMLIT =================
st.set_page_config(page_title="VN Fake News Detector", page_icon="🛡️")
st.title("🛡️ Hệ thống Phát hiện Tin giả tiếng Việt")
st.markdown("---")

input_text = st.text_area("📄 Dán nội dung bài báo cần kiểm tra vào đây:", height=250, placeholder="Ví dụ: Thủ tướng Nhật Bản Shinzo Abe...")

if st.button("🔍 PHÂN TÍCH TIN TỨC"):
    if len(input_text.strip()) < 20:
        st.warning("⚠️ Nội dung bài báo quá ngắn để phân tích chính xác.")
    else:
        with st.spinner("🤖 Đang trích xuất đặc trưng ngôn ngữ PhoBERT..."):
            try:
                probs = predict_news(input_text)
                prob_real, prob_fake = probs[0], probs[1]
                
                st.subheader("📊 Kết quả phân tích:")
                
                if prob_fake > prob_real:
                    st.error(f"### ❌ Cảnh báo: TIN GIẢ ({prob_fake*100:.2f}%)")
                    st.progress(prob_fake)
                else:
                    st.success(f"### ✅ Đánh giá: TIN THẬT ({prob_real*100:.2f}%)")
                    st.progress(prob_real)
                
                # Hiển thị biểu đồ cột cho trực quan
                st.bar_chart({"Tin thật": prob_real, "Tin giả": prob_fake})
                
            except Exception as e:
                st.error(f"💥 Đã xảy ra lỗi: {e}")

st.markdown("---")
st.caption("Mô hình sử dụng PhoBERT + SVM cho Nghiên cứu khoa học. Độ chính xác huấn luyện: 96.77%")