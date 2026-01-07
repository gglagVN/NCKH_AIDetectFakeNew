import os
import pandas as pd
import numpy as np
import torch
import random
import re
import unicodedata
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pyvi import ViTokenizer

# ================= CẤU HÌNH =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Code", "Data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code", "Data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TẢI PHO_BERT =================
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def clean_text_advanced(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================= ĐỌC DỮ LIỆU =================
print("📂 Đang trích xuất dữ liệu...")
fake_list, real_list = [], []

for file_name in os.listdir(RAW_DATA_DIR):
    if file_name.endswith(".csv"):
        path = os.path.join(RAW_DATA_DIR, file_name)
        try:
            # Sửa lỗi encoding_errors
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
            text_col = 'text' if 'text' in df.columns else 'post_message'
            
            for _, row in df.iterrows():
                txt = clean_text_advanced(str(row[text_col]))
                lbl = str(row['label']).split(';')[0].strip().replace('"', '')
                if txt:
                    if lbl == '1': fake_list.append(txt)
                    elif lbl == '0': real_list.append(txt)
        except Exception as e:
            print(f"⚠️ Lỗi file {file_name}: {e}")

num_each = min(len(fake_list), len(real_list))
all_texts = random.sample(fake_list, num_each) + random.sample(real_list, num_each)
all_labels = [1]*num_each + [0]*num_each

# ================= EMBEDDING =================
def get_embeddings(text_list, batch_size=4):
    all_embeddings = []
    model.eval()
    for i in tqdm(range(0, len(text_list), batch_size), desc="PhoBERT Embedding"):
        batch = [ViTokenizer.tokenize(t) for t in text_list[i : i + batch_size]]
        inputs = tokenizer(batch, max_length=256, truncation=True, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

X = get_embeddings(all_texts)
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(all_labels))
print(f"✅ Đã xử lý {X.shape[0]} mẫu dữ liệu.")