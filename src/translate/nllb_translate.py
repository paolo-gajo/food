import ctranslate2
import transformers
from tqdm.auto import tqdm

src_lang = "eng_Latn"
tgt_lang = "ita_Latn"

translator = ctranslate2.Translator('/home/pgajo/nllbct2', 'cuda', compute_type = 'int8_float32')
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=src_lang)

target_prefix = [tgt_lang]

filepath = "/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format_raw.en"

with open(filepath, "r") as f:
    recipes = f.readlines()

translated_recipes = []
for recipe in tqdm(recipes, total=len(recipes)):
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(recipe))
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]
    target_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))
    translated_recipes.append(target_text)
with open(filepath.replace('.en', '_nllb.it'), "w") as f:
    f.writelines([text for text in translated_recipes])