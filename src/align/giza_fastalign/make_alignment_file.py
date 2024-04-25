import json
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import EntityShifter, TASTEset
from tqdm.auto import tqdm
import os

# json_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD_301.json'
json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_unaligned_ls.json'
# json_path = '/home/pgajo/food/data/mycolombianrecipes/MCR-GOLD_291.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

text_tokenized_src = []
text_tokenized_tgt = []

lang_src = 'en'
lang_tgt = 'it'

# data = TASTEset.tasteset_to_label_studio(data, model_name='gold', languages=[lang_src, lang_tgt])

shifter = EntityShifter(languages=[lang_src, lang_tgt])

text_field = 'ingredients'
data_format = 'label_studio'
# data_format = 'tasteset'
strategy = 'regex'

data_tokenized = []
for line in tqdm(data, total=len(data)):
    line_tokenized = shifter.sub_shift(line,
                        text_field = text_field,
                        data_format = data_format,
                        strategy = strategy,
                        # verbose=True,
                        )
    data_tokenized.append(line_tokenized)

new_dir = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

json_path_new = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}.json')
with open(os.path.join(new_dir, os.path.basename(json_path_new)), 'w', encoding='utf8') as f:
    json.dump(data_tokenized, f, ensure_ascii = False)

if data_format == 'label_studio':
    for line in data_tokenized:
        text_tokenized_src.append(line['data'][f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line['data'][f'{text_field}_{lang_tgt}'])
elif data_format == 'tasteset':
    for line in data_tokenized:
        text_tokenized_src.append(line[f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line[f'{text_field}_{lang_tgt}'])

output_filename_en = json_path.replace('.json', f'_tok_{strategy}_{lang_src}.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename_en)), 'w', encoding='utf8') as f:
    for line in text_tokenized_src:
        f.write(line + '\n')

output_filename_it = json_path.replace('.json', f'_tok_{strategy}_{lang_tgt}.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename_it)), 'w', encoding='utf8') as f:
    for line in text_tokenized_tgt:
        f.write(line + '\n')

parallel_list = []

for sent_src, sent_tgt in zip(text_tokenized_src, text_tokenized_tgt):
    parallel_list.append(f'{sent_src} ||| {sent_tgt}')

output_filename = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename)), 'w', encoding='utf8') as f:
    for line in parallel_list:
        f.write(line + '\n')