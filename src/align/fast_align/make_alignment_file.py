import json
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import EntityShifter
from tqdm.auto import tqdm

json_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_unaligned_spaced.json'
# json_path = '/home/pgajo/food/data/mycolombianrecipes/mycolombianrecipes.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

text_tokenized_src = []
text_tokenized_tgt = []

lang_src = 'en'
lang_tgt = 'it'

shifter = EntityShifter(languages = [lang_src, lang_tgt])

text_field = 'ingredients'
data_format = 'label_studio'
strategy = 'moses'

data_tokenized = []
for line in tqdm(data, total=len(data)):
    line_tokenized = shifter.sub_shift(line,
                        text_field = text_field,
                        data_format = data_format,
                        strategy = strategy,
                        verbose=True,
                        )
    data_tokenized.append(line_tokenized)

with open(json_path.replace('.json', f'_tok_{strategy}.json'), 'w', encoding='utf8') as f:
    json.dump(data_tokenized, f, ensure_ascii = False)

if data_format == 'label_studio':
    for line in data_tokenized:
        text_tokenized_src.append(line['data'][f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line['data'][f'{text_field}_{lang_tgt}'])
elif data_format == 'tasteset':
    for line in data_tokenized:
        text_tokenized_src.append(line[f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line[f'{text_field}_{lang_tgt}'])

output_filename_en = json_path.replace('.json', f'_tok_{strategy}_en.txt')
with open(output_filename_en, 'w', encoding='utf8') as f:
    for line in text_tokenized_src:
        f.write(line + '\n')

output_filename_it = json_path.replace('.json', f'_tok_{strategy}_it.txt')
with open(output_filename_it, 'w', encoding='utf8') as f:
    for line in text_tokenized_tgt:
        f.write(line + '\n')

parallel_list = []

for sent_src, sent_tgt in zip(text_tokenized_src, text_tokenized_tgt):
    parallel_list.append(f'{sent_src} ||| {sent_tgt}')

output_filename = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}.txt')
with open(output_filename, 'w', encoding='utf8') as f:
    for line in parallel_list:
        f.write(line + '\n')