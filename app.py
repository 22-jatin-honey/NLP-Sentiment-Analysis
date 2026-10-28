import json
import re
from pathlib import Path

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "bilstm_sentiment.keras"
WORD_INDEX_PATH = BASE_DIR / "artifacts" / "word_index.json"
MAX_LEN = 250
MAX_FEATURES = 30000

st.set_page_config(page_title="ReviewSense | Sentiment AI", page_icon="💬", layout="centered")
st.title("💬 ReviewSense")
st.subheader("Deep Learning Sentiment Analyzer")
st.write("Classify a movie or product review as **positive** or **negative** using a trained Bidirectional LSTM model.")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not WORD_INDEX_PATH.exists():
        return None, None
    model = load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    return model, word_index

def encode_review(text, word_index):
    # Keras IMDB dataset reserves 0=padding, 1=start, 2=unknown, 3=unused.
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    sequence = [1]
    for token in tokens:
        idx = word_index.get(token)
        sequence.append(int(idx) + 3 if idx is not None and int(idx) + 3 < MAX_FEATURES else 2)
    return pad_sequences([sequence], maxlen=MAX_LEN, padding="pre", truncating="pre")

model, word_index = load_artifacts()
if model is None:
    st.warning("The trained model is not available yet. Train it first using the steps below.")
    st.code("pip install -r requirements.txt\npython train.py\nstreamlit run app.py", language="bash")
    st.stop()

review = st.text_area("Paste a review", placeholder="The story was engaging, the acting was excellent, and I loved every minute...", height=160)
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.info("Enter a review to analyze.")
    else:
        x = encode_review(review, word_index)
        probability = float(model.predict(x, verbose=0).ravel()[0])
        label = "Positive 😊" if probability >= 0.5 else "Negative 😞"
        confidence = probability if probability >= 0.5 else 1 - probability
        st.markdown(f"## {label}")
        st.progress(confidence, text=f"Model confidence: {confidence:.1%}")
        st.caption("This is a model prediction, not a guarantee of the review's true sentiment.")

with st.expander("About this project"):
    st.markdown("""
    **Architecture:** Embedding → Bidirectional LSTM → Dense sigmoid output  
    **Task:** Binary sentiment classification  
    **Training data:** IMDB movie-review dataset from Keras  
    **Stack:** Python, TensorFlow/Keras, Streamlit  

    The model learns contextual patterns in sequences of words. A sigmoid score near 1 indicates positive sentiment; a score near 0 indicates negative sentiment.
    """)
