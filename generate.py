from openai import OpenAI
from llm import answer_for_llm, answer_router, answer_critery

import json
import re

def extract_json(text: str):
    # убираем ```json и ```
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


# 1. Инициализация клиента OpenRouter через OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# 2. Формируем prompt через RAG
query = "Яляется ли зацикливание уровней допустимым критерием прохождения игры?"
# query = "считается ли игра пройденной если я дошел до зацикливания уровней?"
# query = "Если в игре получена концовка после которой есть геймплей. Например дполнительный акт или еще сюжет. Можно ли ее зачитывать или нужно проходить дальше"


router_prompt = answer_router(query)

# 3. Отправляем запрос модели
selector = client.chat.completions.create(
    model="openai/gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": router_prompt}]}
    ]
)


raw = selector.choices[0].message.content
data = extract_json(raw)
sections = data["sections"]



if any(s in {1,2,3} for s in sections):
    prompt = answer_critery(query)
    print("Выбор по критерию")
else:
    prompt = answer_for_llm(query)
    print("Другой выбор")

# 3. Отправляем запрос модели
completion = client.chat.completions.create(
    model="openai/gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
)

# 4. Выводим ответ
print(completion.choices[0].message.content)