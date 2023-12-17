import deepl
from tqdm.auto import tqdm
auth_key = "e056a862-0b99-c41d-46a9-57f62ab6cf61:fx"
translator = deepl.Translator(auth_key)

with open("/home/pgajo/working/food/data/TASTEset/data/TASTEset_raw.en", "r") as f:
    recipes = f.readlines()

translated_recipes = []
for recipe in tqdm(recipes, total=len(recipes)): 
    translated_recipe = translator.translate_text(recipe, target_lang="IT")
    translated_recipes.append(translated_recipe)
usage = translator.get_usage()
print(usage.character.count)
with open("/home/pgajo/working/food/data/TASTEset/data/TASTEset_raw.it", "w") as f:
    f.writelines([el.text for el in translated_recipes])