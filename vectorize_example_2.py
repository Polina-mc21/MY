import numpy as np
from sentence_transformers import SentenceTransformer

z = np.load("index.npz", allow_pickle=True)
embs = z["embs"]
texts = z["texts"].tolist()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text = "Мир"
emb = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

sims = embs @ emb

topk = sims.argsort()[-3:][::-1]

print("Top-3 совпадений:")
for rank, i in enumerate(topk[:3], 1):
    print(f"{rank}. idx={int(i)}  score={sims[i]:.4f}\n{texts[i][:200]}...\n---")
