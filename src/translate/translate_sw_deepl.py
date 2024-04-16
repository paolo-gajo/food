import deepl
from tqdm.auto import tqdm
import os
import json

auth_key = os.environ['DEEPL_TOKEN']
translator = deepl.Translator(auth_key)

path_glossary = '/home/pgajo/food/src/translate/glossary_sw.csv'
with open(path_glossary, 'r',  encoding='utf-8') as csv_file:
    csv_data = csv_file.read()  # Read the file contents as a string
    my_csv_glossary = translator.create_glossary_from_csv(
        "CSV glossary",
        source_lang="en",
        target_lang="it",
        csv_data=csv_data,
    )

filepath = "/home/pgajo/food/data/TASTEset/data/formatted/TASTEset_sep_format_spaced_TS.json"
par_dir = os.path.dirname(filepath)
lang_src = 'en'
lang_tgt = 'es'

with open(filepath, 'r', encoding='utf8') as f:
    data = json.load(f)
    
usage = translator.get_usage()
translated_recipes = []
progbar = tqdm(data, total=len(data))
for recipe in progbar: 
    translated_recipe = translator.translate_text(recipe[f'ingredients_{lang_src}'],
                                                  source_lang = lang_src,
                                                  target_lang = lang_tgt,
                                                #   glossary = my_csv_glossary,
                                                  )
    translated_recipes.append(translated_recipe)
    usage = translator.get_usage()
    progbar.set_description(desc=f"DeepL usage: {usage.character.count}")

filepath_out = f'TASTEset_sep_format_raw_{lang_tgt}.txt'

with open(os.path.join(par_dir, filepath_out), "w") as f:
    for el in translated_recipes:
        f.write(el.text + '\n')