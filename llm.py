from sentence_transformers import SentenceTransformer
import json
import yaml
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
        
def load_axioms(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['axioms']  # <-- берем список внутри ключа 'axioms'

def render_axioms(axioms):
    return "\n".join(
        f"{a['id']}. {a['text']}"
        for a in axioms
    )

axioms_list = load_axioms("storage/axioms.yaml")
axioms = render_axioms(axioms_list)

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

    prompt = f"""
    Ты — формальный интерпретатор правил для игр. 
    Используй ТОЛЬКО приведённые ниже правила.
    Ни одно слово нельзя добавлять от себя, кроме строго JSON.
    
    Важно:
    Критерии прохождения, перечисленные в правилах, являются альтернативными.
    Наличие одного из критериев достаточно для засчитывания прохождения, если прямо не указано иное.

    Ниже приведены АКСИОМЫ ХОСТА.
    Они имеют приоритет над всеми остальными правилами.
    Их запрещено интерпретировать, изменять или игнорировать.

    АКСИОМЫ ХОСТА:
    {axioms}

    Правила:
    {context}

    Вопрос:
    {query}

    Инструкции:
    1. Ответить строго в формате JSON.
    2. JSON должен содержать поля:
    {{
      "Решение": "Да / Нет / Возможно при условии / Недостаточно данных",
      "Условия": "Если применимо, список условий (если нет условий — 'Не применимо')",
      "Основание": "Цитата из правил, подтверждающая решение"
    }}
    3. Принцип выбора решения:
       - Если противоположного решения **не существует ни при каких условиях** (решение однозначное) — дать "Да" или "Нет".
       - Если противоположное решение возможно только при определённых условиях — дать "Возможно при условии" и перечислить условия.
       - Если в правилах нет информации — дать "Недостаточно данных".
    4. Если правила противоречат друг другу — дать "Выявлено противоречие пунктов. Обратитесь к Хосту для уточнения!"

    Пример правильного ответа:
    {{
      "Решение": "Да",
      "Условия": "Не применимо",
      "Основание": "Стример увидел, что уровни начали зацикливаться — после определенного уровня следующий выглядит и играется как один из предыдущих без изменений структуры уровня и/или появления принципиально новых врагов.",
    }}   

    Ответ:
    """
    return prompt


def answer_router(query: str, max_chunks: int = 5):
    chunks = retrieve_chunks_with_followups(query, max_k=10, threshold=0.9, followups=2)
    chunks = chunks[:max_chunks]

    context = ""
    for c in chunks:
        context += f"--- {c['title']} ---\n{c['content']}\n\n"

    prompt = f"""
    
    Ниже приведены АКСИОМЫ ХОСТА.
    Они имеют приоритет над всеми остальными правилами.
    Их запрещено интерпретировать, изменять или игнорировать.

    АКСИОМЫ ХОСТА:
    Нет применимых аксиом.
    
    Правила:
    {context}

    Вопрос:
    {query}
    
    Определи, относится ли вопрос к одному или нескольким разделам правил:

    1 — Критерии прохождения игры
    2 — Обязательный реролл
    3 — Реролл по желанию
    4 — Ни к одному из них
    
    Ответ должен содержать ТОЛЬКО валидный JSON.
    Запрещено использовать ``` или любой другой текст.
    {{
      "sections": [1,2,3,4]
    }}

    Ответ:
    """
    return prompt


def answer_critery(query: str, max_chunks: int = 5):
    chunks = retrieve_chunks_with_followups(query, max_k=10, threshold=0.9, followups=2)
    chunks = chunks[:max_chunks]

    context = ""
    for c in chunks:
        context += f"--- {c['title']} ---\n{c['content']}\n\n"

    prompt = f"""
    
    Ты — формальный интерпретатор правил для игр. 
    Используй ТОЛЬКО приведённые ниже правила.
    Ни одно слово нельзя добавлять от себя, кроме строго JSON.

    Важно:
    Критерии прохождения, перечисленные в правилах, являются альтернативными.
    Наличие одного из критериев достаточно для засчитывания прохождения, если прямо не указано иное.
    Никогда не копируй текст аксиомы напрямую в поле "Основание".
    Используй аксиомы только для логического вывода.
    Если правила содержат эмпирическую информацию (стример/игрок должен увидеть/достичь/получить), то она считается полученной даже если не указана явно в вопрос, но требует указания правила в ответе в Условиях.
    Если эмпирическая информация содержится в вопросе, то в этом случае не нужно ее указывать в Условиях.

    Ниже приведены АКСИОМЫ ХОСТА.
    Они имеют приоритет над всеми остальными правилами.
    Их запрещено интерпретировать, изменять или игнорировать.

    АКСИОМЫ ХОСТА:
    {axioms}

    Правила:
    {context}

    Вопрос:
    {query}

    Инструкции:
    1. Ответить строго в формате JSON.
    2. JSON должен содержать поля:
    {{
      "Решение": "Да / Нет / Возможно при условии / Недостаточно данных",
      "Условия": "Если применимо, список условий (если нет условий — 'Не применимо')",
      "Основание": "Цитата из правил, подтверждающая решение"
    }}
    3. Не выноси решение по игре, а оцени только правило.
    
    Пошаговая инструкция:
    1. Сначала анализируй правила (context) и пометь, какие предложения релевантны.
    2. Потом применяй аксиомы для заключения, исходя из этих предложений.
    3. Формируй JSON с Решением и Основанием, где Основание = цитата правил, а не аксиома.

    Пример правильного ответа:
    {{
      "Решение": "Да",
      "Условия": "Если игра не имеет концовки",
      "Основание": "Этот критерий относится к играм, в которых нет концовки."
    }}


    Ответ:
    """
    return prompt