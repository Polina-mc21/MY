# python md_to_vectors.py [tz.md] -> index.npz
import sys, re, numpy as np
from sentence_transformers import SentenceTransformer

FILENAME = sys.argv[1] if len(sys.argv) > 1 else "input.md"

# 1) читаем markdown
with open(FILENAME, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# 2) делим по заголовкам (#..######), заголовок остаётся в тексте куска
chunks = [c.strip() for c in re.split(r'(?m)^(?=#)', text) if c.strip()]

# 3) эмбеддинги (один вектор на раздел)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)

# 4) сохраняем в один файл (вектора + тексты)
np.savez("index.npz", embs=embs, texts=np.array(chunks, dtype=object))

print("Готово. Разделов:", len(chunks), "| Файл: index.npz")
print("Примеры разделов:")
for i in range(min(3, len(chunks))):
    print(f"{i+1}. {chunks[i][:100]}...")
