import streamlit as st
import numpy as np
import re
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =======================
# CONFIG
# =======================
MODEL_CHECKPOINT = "Rifky/indobert-hoax-classification"
BASE_MODEL_CHECKPOINT = "indobenchmark/indobert-base-p1"
DATA_CHECKPOINT = "Rifky/indonesian-hoax-news"

LABEL = {0: "valid", 1: "fake"}

# =======================
# LOAD MODEL (CACHE – OLD STREAMLIT)
# =======================
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, fast=True)
    base_model = SentenceTransformer(BASE_MODEL_CHECKPOINT)
    data = load_dataset(DATA_CHECKPOINT, split="train")
    return model, tokenizer, base_model, data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =======================
# APP START
# =======================
st.markdown(
    "<h1 style='text-align:center;'>Fake News Detection AI</h1>",
    unsafe_allow_html=True
)

with st.spinner("Loading Model..."):
    model, tokenizer, base_model, data = load_model()

# =======================
# INPUT
# =======================
st.info("👉 Paste article text to test (URL scraping is unstable)")

user_text = st.text_area(
    "Article Content",
    height=300,
    placeholder="Paste news article text here..."
)

submit = st.button("Analyze")

# =======================
# RUN
# =======================
if submit:

    if not user_text.strip():
        st.error("Please input article content.")
        st.stop()

    start_time = time.time()

    text = re.sub(r"\n+", " ", user_text)
    tokens = text.split()

    sequences = []
    for i in range(0, len(tokens), 512):
        sequences.append(" ".join(tokens[i:i + 512]))

    inputs = tokenizer(
        sequences,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with st.spinner("Analyzing..."):
        outputs = model(**inputs).logits.detach().numpy()

    fake_score = np.mean([sigmoid(x[1]) for x in outputs])
    valid_score = np.mean([sigmoid(x[0]) for x in outputs])

    prediction = np.argmax([valid_score, fake_score])

    if prediction == 0:
        st.success(f"🟢 This article is **VALID**")
    else:
        st.error(f"🔴 This article is **FAKE**")

    st.markdown(
        f"""
        **Confidence**
        - Valid: `{valid_score:.2%}`
        - Fake: `{fake_score:.2%}`
        """
    )

    # =======================
    # RELATED ARTICLES
    # =======================
    title_embedding = base_model.encode("User Article")

    similarity = cosine_similarity(
        [title_embedding],
        data["embeddings"]
    ).flatten()

    top_idx = np.argsort(similarity)[::-1][:5]

    with st.expander("Related Articles"):
        for idx in top_idx:
            st.markdown(
                f"""
                <small>{data["url"][idx].split('/')[2]}</small><br>
                <a href="{data['url'][idx]}">{data['title'][idx]}</a><br><br>
                """,
                unsafe_allow_html=True
            )

    st.caption(f"⏱️ Time: {time.time() - start_time:.2f}s")
