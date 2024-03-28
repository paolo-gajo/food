import json
import re
import os
from icecream import ic
json_path = '/home/pgajo/food/data/TASTEset/data/formatted/TASTEset_sep_format_spaced_TS.json'
with open(json_path) as json_file:
    data = json.load(json_file)

recipe_list = data['annotations']#[:3]

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
src_lang = 'en'
tgt_lang = 'de'

lang_dict = {'de': 'German'}

from tqdm.auto import tqdm
pbar = tqdm(recipe_list, total=len(recipe_list))
for recipe in pbar: 
    recipe[f'text_{tgt_lang[:2]}'] = ''
    recipe[f'ents_{tgt_lang[:2]}'] = []
    # original_texts = [recipe[f'text_{src_lang[:2]}'][entity[0]:entity[1]] for entity in recipe[f'ents_{src_lang[:2]}']]

    for i, entity in enumerate(recipe[f'ents_{src_lang[:2]}']):
        text_src = recipe[f'text_{src_lang[:2]}'][:entity[0]] + '• ' + recipe[f'text_{src_lang[:2]}'][entity[0]:entity[1]] + ' •' + recipe[f'text_{src_lang[:2]}'][entity[1]:]
        messages = [{"role": "user", "content": f"""Take the text between • markers and translate only that into {lang_dict[tgt_lang]}.
                     \n\n{text_src}"""}]
        
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=400, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0].split('[/INST]')[1]

        # print("original_text", original_text)
        translated_text = ic(re.search(re.compile(r'• (\w+) •'), decoded).group(1))
        # print("translated_text", translated_text)
        recipe[f'text_{tgt_lang[:2]}'] += translated_text
        # print("recipe[f'text_{tgt_lang[:2]}']", recipe[f'text_{tgt_lang[:2]}'])
        recipe[f'ents_{tgt_lang[:2]}'].append([len(recipe[f'text_{tgt_lang[:2]}']) - len(translated_text), len(recipe[f'text_{tgt_lang[:2]}']), entity[2]])
        sep_len = 0
        # find the next alphanumeric character after the entity and put whatever is in between the entity and the next alphanumeric character in the separator
        while entity[1]+sep_len < len(recipe[f'text_{src_lang[:2]}']) and not recipe[f'text_{src_lang[:2]}'][entity[1]+sep_len].isalnum():
            sep_len += 1
        separator = recipe[f'text_{src_lang[:2]}'][entity[1]:entity[1]+sep_len] # use the separator from the original text
        if separator == '':
            separator = ' '
        recipe[f'text_{tgt_lang[:2]}'] += separator 
    recipe[f'text_{tgt_lang[:2]}'] = recipe[f'text_{tgt_lang[:2]}'].strip()
    # print("recipe[f'text_{tgt_lang[:2]}']", recipe[f'text_{tgt_lang[:2]}'])
print(recipe_list)

# check if the file is fine
for i, recipe in enumerate(recipe_list):
    print(i, '------------------')
    print(f"recipe[f'text_{src_lang[:2]}']", recipe[f'text_{src_lang[:2]}'])
    print(f"recipe[f'text_{tgt_lang[:2]}']", recipe[f'text_{tgt_lang[:2]}'])
    for entity in recipe[f'ents_{tgt_lang[:2]}']:
        # print using the original separator
        print(recipe[f'text_{tgt_lang[:2]}'][entity[0]:entity[1]+1].strip(), end = ' ')
    print()

new_data = {'classes': data['classes'], 'annotations': recipe_list}
save_path = os.path.join(os.path.dirname(json_path), f'EW-TT-MT_{src_lang}-{tgt_lang}.json')
with open(save_path, 'w') as json_file:
    json.dump(new_data, json_file)