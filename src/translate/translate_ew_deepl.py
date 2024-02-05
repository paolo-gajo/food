import json
json_path = '/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format.json'
with open(json_path) as json_file:
    data = json.load(json_file)

recipe_list = data['annotations']
import deepl
from tqdm.auto import tqdm
import os
auth_key = os.environ['DEEPL_TOKEN']
translator = deepl.Translator(auth_key)

with open('/home/pgajo/working/food/src/translate/glossary.csv', 'r',  encoding='utf-8') as csv_file:
    csv_data = csv_file.read()  # Read the file contents as a string
    my_csv_glossary = translator.create_glossary_from_csv(
        "CSV glossary",
        source_lang="EN",
        target_lang="IT",
        csv_data=csv_data,
    )

from tqdm.auto import tqdm
pbar = tqdm(recipe_list, total=len(recipe_list))
for recipe in pbar:
    pbar.set_description(f"DeepL usage: {translator.get_usage().character.count}")
    recipe['text_it'] = ''
    recipe['ents_it'] = []
    entity_translations = translator.translate_text([recipe['text_en'][entity[0]:entity[1]] for entity in recipe['ents_en']], 
                                                source_lang = "EN",
                                                target_lang="IT",
                                                glossary = my_csv_glossary,
                                                )
    for i, entity in enumerate(recipe['ents_en']):
        original_text = recipe['text_en'][entity[0]:entity[1]]
        # print("original_text", original_text)
        translated_text = entity_translations[i].text
        # print("translated_text", translated_text)
        recipe['text_it'] += translated_text
        # print("recipe['text_it']", recipe['text_it'])
        recipe['ents_it'].append([len(recipe['text_it']) - len(translated_text), len(recipe['text_it']), entity[2]])
        sep_len = 0
        # find the next alphanumeric character after the entity and put whatever is in between the entity and the next alphanumeric character in the separator
        while entity[1]+sep_len < len(recipe['text_en']) and not recipe['text_en'][entity[1]+sep_len].isalnum():
            sep_len += 1
        separator = recipe['text_en'][entity[1]:entity[1]+sep_len] # use the separator from the original text
        if separator == '':
            separator = ' '
        recipe['text_it'] += separator 
    recipe['text_it'] = recipe['text_it'].strip()
    # print("recipe['text_it']", recipe['text_it'])
print(recipe_list)

# check if the file is fine
for i, recipe in enumerate(recipe_list):
    print(i, '------------------')
    print("recipe['text_en']", recipe['text_en'])
    print("recipe['text_it']", recipe['text_it'])
    for entity in recipe['ents_it']:
        # print using the original separator
        print(recipe['text_it'][entity[0]:entity[1]+1].strip(), end = ' ')
    print()

new_data = {'classes': data['classes'], 'annotations': recipe_list}

with open(json_path.replace('.json', '_en-it_DEEPL_glossary.json'), 'w') as json_file:
    json.dump(new_data, json_file)