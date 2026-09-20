import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

STYLE_GUIDE = "anime_style_guide.txt"
MODEL = "phi3:mini"
USE_RAG = True
THRESHOLD = 0.1  # Lowered

embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open(STYLE_GUIDE) as f:
    chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
embeddings = embedder.encode(chunks)

def get_context(question):
    q_emb = embedder.encode([question])
    scores = np.dot(embeddings, q_emb.T).flatten()
    top = np.argsort(scores)[-3:][::-1]
    # print(f"  [DEBUG] Top scores: {[round(scores[i], 3) for i in top]}")
    relevant = [chunks[i] for i in top if scores[i] > THRESHOLD]
    return "\n\n".join(relevant) if relevant else None

history = []
print(f"RAG: {'ON' if USE_RAG else 'OFF'} | Ctrl+C to quit\n")

while True:
    question = input("You: ")
    context = get_context(question) if USE_RAG else None

    if context:
        system = """You are a helpful assistant with one strict rule:
Every single response MUST include at least one anime reference.
Connect it naturally to whatever you are explaining.
Use characters and moments from: Naruto, Attack on Titan, Dragon Ball Z, My Hero Academia, Death Note, Fullmetal Alchemist.

Style guide:
""" + context
    else:
        system = "You are a helpful assistant."

    history.append({"role": "user", "content": question})
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + history
    )
    answer = response["message"]["content"]
    history.append({"role": "assistant", "content": answer})
    print(f"\nModel: {answer}\n")