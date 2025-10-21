# AI Manuals Chatbot (Streamlit App)

This project lets you upload machinery or appliance manuals (PDFs), automatically extract technical parameters, and ask natural-language questions like:

> “Compare viscosity and flash point across brands.”

---

## 🧰 Features
- PDF → text + OCR fallback
- Parameter extraction (Viscosity, Flash Point, Density, etc.)
- FAISS + BM25 retrieval for contextual search
- FLAN-T5 small LLM for concise, grounded answers
- Streamlit-based interactive UI

---

## 🏗️ Installation

```bash
git clone https://github.com/<your-username>/ai_manuals_chatbot.git
cd ai_manuals_chatbot
pip install -r requirements.txt
streamlit run app.py
