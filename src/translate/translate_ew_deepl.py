import json
from icecream import ic
json_path = '/home/pgajo/food/data/TASTEset/data/formatted/TASTEset_sep_format_spaced_TS.json'
with open(json_path) as json_file:
    data = json.load(json_file)

recipe_list = data['annotations']#[:3]

import deepl
from tqdm.auto import tqdm
import os
auth_key = os.environ['DEEPL_TOKEN']
translator = deepl.Translator(auth_key)

src_lang = 'en'
tgt_lang = 'es'

from tqdm.auto import tqdm
pbar = tqdm(recipe_list, total=len(recipe_list))
for recipe in pbar:
    pbar.set_description(f"DeepL usage: {translator.get_usage().character.count}")
    recipe[f'text_{tgt_lang}'] = ''
    recipe[f'ents_{tgt_lang}'] = []

    for i, entity in enumerate(recipe[f'ents_{src_lang}']):
        original_text = recipe[f'text_{src_lang}'][entity[0]:entity[1]]
        # print("original_text", original_text)
        translated_text = translator.translate_text(original_text,
                                                    source_lang = src_lang,
                                                    target_lang = tgt_lang,
                                                    context=recipe[f'text_{src_lang}']).text
        recipe[f'text_{tgt_lang}'] += ic(translated_text)
        # print("recipe[f'text_{tgt_lang}']", recipe[f'text_{tgt_lang}'])
        recipe[f'ents_{tgt_lang}'].append([len(recipe[f'text_{tgt_lang}']) - len(translated_text), len(recipe[f'text_{tgt_lang}']), entity[2]])
        sep_len = 0
        # find the next alphanumeric character after the entity and put whatever is in between the entity and the next alphanumeric character in the separator
        while entity[1]+sep_len < len(recipe[f'text_{src_lang}']) and not recipe[f'text_{src_lang}'][entity[1]+sep_len].isalnum():
            sep_len += 1
        separator = recipe[f'text_{src_lang}'][entity[1]:entity[1]+sep_len] # use the separator from the original text
        if separator == '':
            separator = ' '
        recipe[f'text_{tgt_lang}'] += separator 
    recipe[f'text_{tgt_lang}'] = recipe[f'text_{tgt_lang}'].strip()
    # print("recipe[f'text_{tgt_lang}']", recipe[f'text_{tgt_lang}'])
print(recipe_list)

# check if the file is fine
for i, recipe in enumerate(recipe_list):
    print(i, '------------------')
    print(f"recipe[f'text_{src_lang}']", recipe[f'text_{src_lang}'])
    print(f"recipe[f'text_{tgt_lang}']", recipe[f'text_{tgt_lang}'])
    for entity in recipe[f'ents_{tgt_lang}']:
        # print using the original separator
        print(recipe[f'text_{tgt_lang}'][entity[0]:entity[1]+1].strip(), end = ' ')
    print()

new_data = {'classes': data['classes'], 'annotations': recipe_list}
save_path = os.path.join(os.path.dirname(json_path), f'EW-TT-MT_{src_lang}-{tgt_lang}_context.json')
with open(save_path, 'w') as json_file:
    json.dump(new_data, json_file)