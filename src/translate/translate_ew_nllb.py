import json
import re
import os
from icecream import ic
json_path = '/home/pgajo/food/data/TASTEset/data/formatted/TASTEset_sep_format.json'
with open(json_path) as json_file:
    data = json.load(json_file)

recipe_list = data['annotations']#[:3]

model_name = 'facebook/nllb-200-3.3B'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='cuda')
src_lang = 'eng_Latn'
tgt_lang = 'deu_Latn'
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
# print(help(model.generate))

def translate_text(intro, text, outro):
    # Tokenize and translate the text
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: inputs[k].to('cuda') for k in inputs.keys()}
    #manual fix to hf bug 
    # inputs['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]
    
    generated_tokens = model.generate(**inputs, max_length=30, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])

    # Decode and return the translated text
    return tokenizer.decode(generated_tokens[0])#, skip_special_tokens=True)

from tqdm.auto import tqdm
pbar = tqdm(recipe_list, total=len(recipe_list))
for recipe in pbar: 
    recipe[f'text_{tgt_lang[:2]}'] = ''
    recipe[f'ents_{tgt_lang[:2]}'] = []
    # original_texts = [recipe[f'text_{src_lang[:2]}'][entity[0]:entity[1]] for entity in recipe[f'ents_{src_lang[:2]}']]

    for i, entity in enumerate(recipe[f'ents_{src_lang[:2]}']):
        intro = recipe[f'text_{src_lang[:2]}'][:entity[0]]
        text_src = recipe[f'text_{src_lang[:2]}'][entity[0]:entity[1]]
        outro = recipe[f'text_{src_lang[:2]}'][entity[1]:]
        text_tgt = translate_text(intro, text_src, outro)
        original_text = recipe[f'text_{src_lang[:2]}'][entity[0]:entity[1]]
        ic(original_text)
        translated_text = ic(re.search(re.compile(r'# (\w+) ###'), text_tgt).group(1))
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