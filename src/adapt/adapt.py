from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

import json
json_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced.json'
with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

lang = 'en'
sample = data[0]['data']
recipe_text = sample[f'title_{lang}'] + '\n\n' + sample[f'presentation_{lang}'] + '\n\n' + sample[f'ingredients_{lang}'] + '\n\n' + sample[f'preparation_{lang}']

messages = [
    {"role": "user", "content": f"Make this recipe vegetarian:\n\n{recipe_text}"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
