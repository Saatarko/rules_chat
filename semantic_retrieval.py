from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


# Load FAISS index
index = faiss.read_index("storage/rules.index")

# Load metadata
metas = []
with open("storage/rules_meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metas.append(json.loads(line))

print("Vectors in index:", index.ntotal)
print("Metadata entries:", len(metas))

model = SentenceTransformer("BAAI/bge-m3")  # Та же модель, что и при embeddings

def retrieve_chunks(query: str, top_k: int = 5):
    # query → embedding
    q_emb = model.encode([query], normalize_embeddings=True)

    # FAISS search
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)

    # Возврат chunks с метаданными
    results = []
    for idx in I[0]:
        meta = metas[idx]
        results.append({
            "title": meta["title"],
            "content": meta.get("content", ""),  # если добавим в мета контент
            "source": meta["source"]
        })
    return results


def retrieve_and_group(query: str, top_k: int = 10):
    chunks = retrieve_chunks(query, top_k=top_k)

    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["title"]].append(c)

    # Сортируем по chunk_index и объединяем content
    results = []
    for title, items in grouped.items():
        items.sort(key=lambda x: x.get("chunk_index", 0))
        combined_text = "\n".join([i["content"] for i in items])
        results.append({
            "title": title,
            "content": combined_text,
            "source": items[0]["source"]
        })
    return results


def retrieve_chunks_with_followups(query: str, max_k: int = 10, threshold: float = 0.9, followups: int = 2):
    # embedding запроса
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), max_k)

    top1_sim = D[0][0]
    results = []
    selected_indices = set()

    # сначала top_chunks по similarity
    for score, idx in zip(D[0], I[0]):
        if score >= top1_sim * threshold:
            results.append(metas[idx])
            selected_indices.add(idx)

    # затем добавляем следующие followups чанки того же parent_id
    for r in list(results):
        parent = r["parent_id"]
        current_idx = r.get("chunk_index", 0)
        for f in range(1, followups + 1):
            next_idx = current_idx + f
            # ищем такой chunk с тем же parent_id
            for i, m in enumerate(metas):
                if m["parent_id"] == parent and m.get("chunk_index", -1) == next_idx and i not in selected_indices:
                    results.append(m)
                    selected_indices.add(i)

    # группируем по title и сортируем по chunk_index
    from collections import defaultdict
    grouped = defaultdict(list)
    for c in results:
        grouped[c["title"]].append(c)

    final_results = []
    for title, items in grouped.items():
        items.sort(key=lambda x: x.get("chunk_index", 0))
        combined_text = "\n".join([i["content"] for i in items])
        final_results.append({
            "title": title,
            "content": combined_text,
            "source": items[0]["source"]
        })

    return final_results


def answer(query):


    results = retrieve_chunks_with_followups(
        query,
        max_k=10,
        threshold=0.9,
        followups=2
    )

    for r in results:
        print(f"--- {r['title']} ---")
        print(r['content'][:1500], "...\n")

def answer_for_llm(query, max_k=5, threshold=0.9, followups=2):
    """
    Возвращает текст для LLM: объединённые chunks по заголовкам.
    """
    # Получаем top-k + followups
    llm_texts = retrieve_chunks_with_followups(
        query,
        max_k=max_k,
        threshold=threshold,
        followups=followups
    )

    # Формируем список для LLM
    return [
        {
            "title": c["title"],
            "text": c["content"],
            "source": c["source"]
        }
        for c in llm_texts
    ]
#
# llm_inputs = answer_for_llm("Яляется ли зацикливание уровней допустимым окончанием игры?")
#
# for item in llm_inputs:
#     print(f"--- {item['title']} ---")
#     print(item['text'][:2000], "...\n")  # первые 2000 символов для просмотра