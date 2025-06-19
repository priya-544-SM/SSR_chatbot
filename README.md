📄 Document Question Answering Chatbot (FastAPI)

This is a FastAPI-based backend for a chatbot that answers questions from a document (`.pdf`, `.docx`, or `.txt`). It uses semantic search with Sentence Transformers and a pre-trained BERT QA model to provide accurate, context-aware answers.

## Features
- Semantic search using Sentence-BERT + FAISS
- Question Answering using HuggingFace's DistilBERT (`distilbert-base-cased-distilled-squad`)
- Supports `.txt`, `.pdf`, and `.docx` documents
- Retrieves top relevant text chunks to answer user queries
- FastAPI backend API

---

## 📁 Project Structure
project/
│
├── main.py # Main FastAPI application
├── document.txt # Sample document (can be PDF, DOCX, or TXT)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

1. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate on Windows # or source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Running the App
- Place your document in the project folder (name it document.txt, document.pdf, or document.docx).

- Run the FastAPI app: 
uvicorn main:app --reload

- Open your browser and go to:
http://127.0.0.1:8000/docs

- Use the /chat POST endpoint to ask questions about your document.

## How it Works
- Document Loading: The document is read and split into smaller chunks.
- Embedding: Chunks are converted into dense vectors using all-MiniLM-L6-v2.
- Indexing: Embeddings are stored in a FAISS index for fast semantic search.
- Query Processing:
User question is embedded and compared with chunks in FAISS.
Top-matching chunks are passed as context to a BERT QA model.
The model predicts the most likely answer span.
- Answer Returned: The best answer is sent back to the client.

## Improvements
 1. Use different Instruct model for better accuracy
 2. Add Yes/No classifier layer
 3. Add support for file uploads
 4. Add frontend UI (React/HTML)
