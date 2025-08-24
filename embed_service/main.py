from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("intfloat/multilingual-e5-small")  # 384d

class Q(BaseModel):
    text: str

@app.post("/embed")
def embed(q: Q):
    v = model.encode([q.text], normalize_embeddings=True)[0].tolist()
    return {"embedding": v}