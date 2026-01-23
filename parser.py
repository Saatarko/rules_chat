import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import unquote

url = "https://rgg.land/rules"
resp = requests.get(url)
resp.encoding = "utf-8"
soup = BeautifulSoup(resp.text, "html.parser")

toc = None
for ul in soup.find_all("ul"):
    if ul.find("a", href=lambda x: x and x.startswith("#")):
        toc = ul
        break

links = toc.find_all("a")


def extract_section(start_tag):
    content = []
    for sibling in start_tag.find_next_siblings():
        if sibling.name and sibling.name.startswith("h"):
            break
        content.append(sibling.get_text(" ", strip=True))
    return "\n".join(content)

docs = []

for i, link in enumerate(links):
    anchor = unquote(link["href"].lstrip("#"))
    header = soup.find(id=anchor)

    if not header:
        continue

    text = extract_section(header)

    docs.append({
        "id": f"rules_{anchor}",
        "type": "base_rule",
        "title": link.text.strip(),
        "content": text,
        "order": i + 1,
        "source": url
    })

with open("docs/rgg_rules.jsonl", "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
