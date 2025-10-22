import os, json, re, hashlib
import streamlit as st
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pdfplumber, pytesseract
from pdf2image import convert_from_path
from unidecode import unidecode

# ===============================================
# Streamlit setup
# ===============================================
st.set_page_config(page_title="AI Manuals Chatbot", layout="wide")
st.title("🧠 AI-Powered Machinery Manual Chatbot")
st.markdown("Upload manuals (PDFs), extract specs, and ask natural questions about them.")

# ===============================================
# Directory setup
# ===============================================
uploaded_files = st.sidebar.file_uploader("Upload PDF Manuals", type=["pdf"], accept_multiple_files=True)
manual_dir = "data/manuals"
os.makedirs(manual_dir, exist_ok=True)
text_cache = "data/text_cache.jsonl"
out_dir = "data/outputs"
os.makedirs(out_dir, exist_ok=True)

# ===============================================
# PDF → Text Extraction (with OCR fallback)
# ===============================================
def read_pdf_text(pdf_file):
    pages_text = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages_text.append((i, unidecode(txt)))

    # OCR fallback if text extraction is too light
    if sum(len(t) for _, t in pages_text) < 400:
        images = convert_from_path(pdf_file, dpi=150)
        pages_text = [(i+1, unidecode(pytesseract.image_to_string(im))) for i, im in enumerate(images)]

    clean = []
    for pnum, t in pages_text:
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{2,}", "\n", t)
        clean.append((pnum, t.strip()))
    return clean

if uploaded_files:
    with open(text_cache, "w") as cache:
        for f in uploaded_files:
            path = os.path.join(manual_dir, f.name)
            with open(path, "wb") as fp:
                fp.write(f.getbuffer())
            st.sidebar.success(f"✅ Saved {f.name}")

            for pnum, txt in read_pdf_text(path):
                rec = {"file": f.name, "page": pnum, "text": txt}
                cache.write(json.dumps(rec) + "\n")

    st.sidebar.success("✅ Text extracted and cached!")

# ===============================================
# Load Extracted Data
# ===============================================
texts, meta = [], []
if os.path.exists(text_cache):
    with open(text_cache) as f:
        for line in f:
            rec = json.loads(line)
            txt = rec["text"].strip()
            if len(txt) > 50:
                texts.append(txt)
                meta.append(f"{rec['file']} p.{rec['page']}")

if len(texts) == 0:
    st.warning("Please upload manuals first.")
    st.stop()

# ===============================================
# Build Retrieval Index (scikit-learn + BM25)
# ===============================================
st.sidebar.write("🔍 Building retrieval index...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
embs = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
bm25 = BM25Okapi([t.split() for t in texts])

# ===============================================
# Load LLM Model (FLAN-T5)
# ===============================================
tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tok)

# ===============================================
# Helper: Retrieval + Answer Synthesis
# ===============================================
def retrieve(query, top_k=5):
    """Retrieve top_k most similar text chunks using cosine similarity + BM25."""
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    sem_hits = [(texts[i], meta[i]) for i in top_idx]

    bm_hits = bm25.get_top_n(query.split(), list(range(len(texts))), n=top_k)
    bm_hits = [(texts[i], meta[i]) for i in bm_hits]

    # Merge results while removing duplicates
    seen, out = set(), []
    for t, m in sem_hits + bm_hits:
        key = hashlib.md5((m + t).encode()).hexdigest()
        if key not in seen:
            out.append((t, m))
            seen.add(key)
        if len(out) >= top_k:
            break
    return out

def generate_answer(question):
    """Generate grounded answers using retrieved context."""
    ctxs = retrieve(question)
    context_text = "\n\n---\n\n".join([f"{m}: {t[:600]}" for t, m in ctxs])
    prompt = f"""You are a technical assistant answering only from manuals.

Question: {question}

Context:
{context_text}

Answer in 3–5 sentences. Mention brands or specs when available."""
    result = generator(prompt, max_new_tokens=200, temperature=0.0)[0]["generated_text"]
    return result.strip()

# ===============================================
# Chat Interface
# ===============================================
st.markdown("### 💬 Ask a Question")
user_query = st.text_input("Enter your question:")
if st.button("Ask") and user_query.strip():
    with st.spinner("Thinking..."):
        answer = generate_answer(user_query)
        st.success(answer)
