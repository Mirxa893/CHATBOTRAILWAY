from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Load RAG files
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

index = faiss.read_index("faiss_index.bin")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ENV
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Utility: get top-k relevant context
def get_context(query, top_k=5):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n".join([texts[i] for i in I[0]])

# Schema for API input
class ChatRequest(BaseModel):
    message: str
    history: list[list[str]] = []

@app.post("/respond")
async def respond(req: ChatRequest):
    context = get_context(req.message)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer: " + context}
    ]

    for user, assistant in req.history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": MODEL,
        "messages": messages
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}"}

@app.post("/upload")
async def upload_txt(file: UploadFile = File(...)):
    try:
        new_texts = [line.strip() for line in (await file.read()).decode('utf-8').splitlines() if line.strip()]
        if not new_texts:
            return {"status": "❌ Uploaded file is empty."}

        # Update JSON
        with open("texts.json", "r", encoding="utf-8") as f:
            current_texts = json.load(f)
        current_texts.extend(new_texts)
        with open("texts.json", "w", encoding="utf-8") as f:
            json.dump(current_texts, f, ensure_ascii=False, indent=2)

        # Update FAISS
        new_vecs = embed_model.encode(new_texts)
        index.add(np.array(new_vecs))
        faiss.write_index(index, "faiss_index.bin")

        return {"status": f"✅ Added {len(new_texts)} new texts."}
    except Exception as e:
        return {"status": f"❌ Error: {str(e)}"}
