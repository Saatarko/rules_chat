import json
import tiktoken
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 500
OVERLAP_TOKENS = 100


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def chunk_text(text: str):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        candidate = current + "\n" + p if current else p
        if count_tokens(candidate) <= TARGET_TOKENS:
            current = candidate
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    # overlap
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        prev_tokens = enc.encode(chunks[i - 1])
        overlap = enc.decode(prev_tokens[-OVERLAP_TOKENS:])
        final_chunks.append(overlap + "\n" + chunk)

    return final_chunks

input_path = Path("docs/rgg_rules.jsonl")
output_path = Path("docs/rgg_rules_chunks.jsonl")

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        doc = json.loads(line)
        chunks = chunk_text(doc["content"])

        for idx, chunk in enumerate(chunks):
            out = {
                "id": f"{doc['id']}_{idx}",
                "parent_id": doc["id"],
                "chunk_index": idx,
                "title": doc["title"],
                "content": chunk,
                "source": doc["source"]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")