from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llm import answer_for_llm
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn

import os

os.environ["HF_HOME"] = "/media/bigdisk/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/media/bigdisk/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/media/bigdisk/hf_cache/transformers"

# 1. Загружаем модель и токенизатор (локально)
model_name = "togethercomputer/RedPajama-INCITE-7B-Instruct"  # пример
cache_dir = "/media/bigdisk/hf_cache/redpajama7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # 🔑 ВАЖНО: декодируем
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# 3. Формируем prompt через нашу функцию
query = "Яляется ли зацикливание уровней допустимым окончанием игры?"
prompt = answer_for_llm(query)

# 4. Генерация ответа
output = generate_answer(prompt)
print(output)