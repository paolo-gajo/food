import deepl
from tqdm.auto import tqdm
import os
auth_key = os.environ['DEEPL_TOKEN']
translator = deepl.Translator(auth_key)

with open('/home/pgajo/working/food/src/translate/glossary_sw.csv', 'r',  encoding='utf-8') as csv_file:
    csv_data = csv_file.read()  # Read the file contents as a string
    my_csv_glossary = translator.create_glossary_from_csv(
        "CSV glossary",
        source_lang="EN",
        target_lang="IT",
        csv_data=csv_data,
    )

filepath = "/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format_raw.en"

with open(filepath, "r") as f:
    recipes = f.readlines()[:10]

translated_recipes = []
for recipe in tqdm(recipes, total=len(recipes)): 
    translated_recipe = translator.translate_text(recipe,
                                                  source_lang = "EN",
                                                  target_lang="IT",
                                                  glossary = my_csv_glossary,
                                                  )
    translated_recipes.append(translated_recipe)
usage = translator.get_usage()
print(usage.character.count)
with open(filepath.replace('.en', 'deepl_glossary.it'), "w") as f:
    f.writelines([el.text for el in translated_recipes])