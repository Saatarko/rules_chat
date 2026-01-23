from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import json
from pathlib import Path

from tqdm import tqdm

model = SentenceTransformer("BAAI/bge-m3")

texts = []
metas = []

with open("docs/rgg_rules_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        texts.append(f"{doc['title']}\n{doc['content']}")
        metas.append(doc)


BATCH_SIZE = 32
all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
    batch_texts = texts[i:i + BATCH_SIZE]

    batch_emb = model.encode(
        batch_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.append(batch_emb)

embeddings = np.vstack(all_embeddings)

DIM = embeddings.shape[1]

index = faiss.IndexFlatIP(DIM)  # cosine similarity (при normalize_embeddings=True)
index.add(np.asarray(embeddings, dtype=np.float32))

Path("storage").mkdir(parents=True, exist_ok=True)

faiss.write_index(index, "storage/rules.index")

with open("storage/rules_meta.jsonl", "w", encoding="utf-8") as f:
    for i, meta in enumerate(metas):
        meta_out = {
            "vector_id": i,
            "id": meta["id"],
            "parent_id": meta["parent_id"],
            "chunk_index": meta["chunk_index"],
            "title": meta["title"],
            "source": meta["source"],
            "content": meta["content"]  # <--- добавляем сюда текст
        }
        f.write(json.dumps(meta_out, ensure_ascii=False) + "\n")
