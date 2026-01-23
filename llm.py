from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# --- 1. Загрузка FAISS индекса и метаданных ---
index = faiss.read_index("storage/rules.index")

metas = []
with open("storage/rules_meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metas.append(json.loads(line))

model = SentenceTransformer("BAAI/bge-m3")  # модель, с которой делались embeddings

# --- 2. Функция поиска с followups ---
def retrieve_chunks_with_followups(query: str, max_k: int = 10, threshold: float = 0.9, followups: int = 2):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), max_k)

    top1_sim = D[0][0]
    results = []
    selected_indices = set()

    # выбираем top chunks по similarity
    for score, idx in zip(D[0], I[0]):
        if score >= top1_sim * threshold:
            results.append(metas[idx])
            selected_indices.add(idx)

    # добавляем follow-up chunks того же parent_id
    for r in list(results):
        parent = r.get("parent_id")
        current_idx = r.get("chunk_index", 0)
        for f in range(1, followups + 1):
            next_idx = current_idx + f
            for i, m in enumerate(metas):
                if m.get("parent_id") == parent and m.get("chunk_index", -1) == next_idx and i not in selected_indices:
                    results.append(m)
                    selected_indices.add(i)

    # Группируем по title и сортируем по chunk_index
    from collections import defaultdict
    grouped = defaultdict(list)
    for c in results:
        grouped[c["title"]].append(c)

    final_results = []
    for title, items in grouped.items():
        items.sort(key=lambda x: x.get("chunk_index", 0))

        # убираем дубликаты предложений
        seen = set()
        clean_sentences = []
        for i in items:
            content = i.get("content", "")
            for s in content.split("\n"):
                s_clean = s.strip()
                if s_clean and s_clean not in seen:
                    seen.add(s_clean)
                    clean_sentences.append(s_clean)
        combined_text = "\n".join(clean_sentences)

        final_results.append({
            "title": title,
            "content": combined_text,
            "source": items[0].get("source", "")
        })

    return final_results


def answer_for_llm(query: str, max_chunks: int = 5):
    chunks = retrieve_chunks_with_followups(query, max_k=10, threshold=0.9, followups=2)
    chunks = chunks[:max_chunks]

    context = ""
    for c in chunks:
        context += f"--- {c['title']} ---\n{c['content']}\n\n"

    prompt = f"""Согласно правилам:

    {context}
    
    Вопрос: {query}
    Ответ:"""
    return prompt

# # --- 4. Проверяем вывод prompt ---
# query = "Является ли зацикливание уровней допустимым окончанием игры?"
# prompt_for_llm = answer_for_llm(query)
# print(prompt_for_llm)