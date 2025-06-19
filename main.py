from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
import docx
import re
# Initialize FastAPI app
app = FastAPI()

# Load document
def extract_pdf_text(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_docx_text(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_txt_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".txt":
        return extract_txt_text(file_path)
    else:
        raise ValueError("Unsupported file type.")

# General Responses Dictionary
general_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! Ask me anything.",
    "hey": "Hey! How can I help you?",
    "how are you": "I'm just a bot, but I'm functioning perfectly! How about you?",
    "what's your name": "I'm a document-based chatbot.",
    "who created you": "I was created by a developer using Python and AI models!",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! If you have more questions, feel free to ask.",
    "thank you": "You're welcome! If you have more questions, feel free to ask.",
    "what can you do": "I can answer questions based on the content of a document you provide. Just ask me anything related to that document!",
    "can you help me": "Of course! Just ask your question, and I'll do my best to help you.",
    "tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
}


# Chunking
def chunk_text(text, max_length=500):
    paragraphs = text.split("\n")
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) < max_length:
            chunk += para + " "
        else:
            chunks.append(chunk.strip())
            chunk = para + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Embedding
def embed_chunks(chunks, embedder):
    return embedder.encode(chunks, convert_to_tensor=False)

# FAISS Index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# Retrieve top chunks
def retrieve(query, chunks, embedder, index, k=5):
    query_vec = embedder.encode([query])[0]
    distances, indices = index.search(np.array([query_vec]).astype("float32"), k)
    return [chunks[i] for i in indices[0]]

# QA Answer
def answer_question(question, context, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))

    if answer.strip() == "" or "[CLS]" in answer or len(answer.split()) < 2:
        return "Sorry, I don't have an answer for that."
    
    return answer

# Load and prepare once at app startup
doc_path = "document.txt"  # Or any supported file
document = load_document(doc_path)
chunks = chunk_text(document)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_chunks(chunks, embedder)
index = build_faiss_index(embeddings)

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Request schema
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    raw_question = request.question
    question = re.sub(r'[^\w\s]', '', raw_question.lower().strip())  # Normalize input

    # question = request.question
    # Check for general response
    if question in general_responses:
        return {"answer": general_responses[question]}

    # Document-based QA
    top_chunks = retrieve(raw_question, chunks, embedder, index)
    context = " ".join(top_chunks)
    answer = answer_question(question, context, model, tokenizer)
    return {"answer": answer}
