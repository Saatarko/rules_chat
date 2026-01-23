from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import json
from pathlib import Path



index = faiss.read_index("storage/rules.index")

metas = []
with open("storage/rules_meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metas.append(json.loads(line))

print(index.ntotal, len(metas))