from openai import OpenAI
from llm import answer_for_llm, answer_router, answer_critery
from dotenv import load_dotenv
import json
import re
import os

def extract_json(text: str):
    # убираем ```json и ```
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


from openai import OpenAI
load_dotenv()  # читает .env

# 1. Инициализация клиента OpenRouter через OpenAI SDK

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
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