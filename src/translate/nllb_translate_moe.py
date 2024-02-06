from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = '/models/huggingface/hub/nllb-moe-54b/'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

filepath = "/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format_raw.en"

with open(filepath, "r") as f:
    recipes = f.readlines()[:5]

translated_recipes = []
for recipe in tqdm(recipes, total=len(recipes)):
    inputs = tokenizer(recipe, return_tensors="pt", padding = True)

    translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"]
    )
    translated_recipes.append(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True))


with open(filepath.replace('.en', '_nllb.it'), "w") as f:
    f.writelines([text for text in translated_recipes])