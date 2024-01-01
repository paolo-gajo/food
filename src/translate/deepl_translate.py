import deepl
from tqdm.auto import tqdm
auth_key = "c8fe40ff-eada-719c-616f-ee298486d413:fx"
translator = deepl.Translator(auth_key)

filepath = "/home/pgajo/working/food/data/TASTEset/data/TASTEset_semicolon_formatted_raw.en"

with open(filepath, "r") as f:
    recipes = f.readlines()

translated_recipes = []
for recipe in tqdm(recipes, total=len(recipes)): 
    translated_recipe = translator.translate_text(recipe, target_lang="IT")
    translated_recipes.append(translated_recipe)
usage = translator.get_usage()
print(usage.character.count)
with open(filepath.replace('.en', '.it'), "w") as f:
    f.writelines([el.text for el in translated_recipes])