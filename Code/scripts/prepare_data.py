import os
import pandas as pd
import numpy as np
import torch
import re
import unicodedata
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pyvi import ViTokenizer

# CẤU HÌNH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Code", "Data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code", "Data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TẢI PHO_BERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

def clean_standard(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_sw(text):
    sw = ["là", "của", "những", "cái", "rằng", "thì", "mà", "vậy", "với", "cho"]
    return " ".join([w for w in text.split() if w not in sw])

print("📂 Đang nạp dữ liệu từ CSV...")
csv_path = os.path.join(RAW_DATA_DIR, "vn_news_226_tlfr.csv")

# Đọc file với cơ chế quotechar để xử lý bài báo nhiều dòng
df = pd.read_csv(csv_path, encoding='utf-8', sep=',', quotechar='"', engine='python', on_bad_lines='skip')

texts, labels = [], []
for _, row in df.iterrows():
    t = clean_standard(str(row['text']))
    t = ViTokenizer.tokenize(t)
    t = remove_sw(t)
    try:
        # Ép kiểu nhãn về số nguyên (0 hoặc 1)
        lbl = int(float(str(row['label']).strip()))
        if t:
            texts.append(t)
            labels.append(lbl)
    except: continue

def get_embeddings(text_list):
    all_emb = []
    model.eval()
    for i in tqdm(range(0, len(text_list), 4), desc="Embedding"):
        batch = text_list[i : i + 4]
        inputs = tokenizer(batch, max_length=256, truncation=True, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            all_emb.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(all_emb, axis=0)

X = get_embeddings(texts)
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(labels))
print(f"✅ Đã xử lý {len(texts)} mẫu.")