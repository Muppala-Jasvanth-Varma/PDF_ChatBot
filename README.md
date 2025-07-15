# 📄 AI-Powered PDF Q&A Web

An intelligent, open-access web application that allows users to upload a PDF and interact with it through natural language queries. This chatbot can answer questions like "What are the skills mentioned?" or "What is the name of the person in the pdf?" using advanced NLP techniques.

---

## 🚀 Features

- ✅ Upload a pdf
- ✅ Ask context-aware questions about the uploaded resume
- ✅ Get instant and accurate answers using AI
- ✅ Built with **LangChain**, **FAISS**, and **Google AI Models**
- ✅ Modern and responsive frontend
- ✅ Serverless or local deployment ready

---

## 🧠 Tech Stack

| Layer        | Technology Used                          |
|-------------|-------------------------------------------|
| Frontend     | HTML, CSS, JavaScript (or your frontend framework) |
| Backend      | Python, Flask or FastAPI                 |
| NLP Engine   | LangChain + Google PaLM / Gemini Pro     |
| Vector DB    | FAISS (for semantic search)              |
| PDF Parsing  | PyPDF2 / pdfplumber                      |
| Hosting  | |   Streamlit

---

## 📂 Project Structure

```bash
resume-chatbot/
│
├── app.py                     # Main backend logic (Flask/FastAPI)
├── utils/
│   ├── extract_text.py        # PDF text extraction logic
│   ├── generate_embeddings.py # Embedding creation using Google AI
│   ├── query_handler.py       # LangChain + FAISS logic
│
├── templates/
│   └── index.html             # Frontend HTML page
│
├── static/
│   ├── style.css              # Optional styling
│   └── script.js              # JS logic (optional)
│
├── sample-resume.pdf          # Demo resume (for testing)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
