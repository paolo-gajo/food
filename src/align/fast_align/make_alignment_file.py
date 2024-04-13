from nltk.tokenize import word_tokenize
import os
import json
import sys
sys.path.append('/home/pgajo/food/data/utils_data')
from subshift import EntityShifter

json_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_TS.json'
with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

text_en_tokenized = []
text_it_tokenized = []

src_lang = 'en'
tgt_lang = 'it'
shifter = EntityShifter()
data = shifter.sub_shift(json_data=data, text_field = 'ingredients', ent_field = 'ents', lang = src_lang)
data = shifter.sub_shift(json_data=data, text_field = 'ingredients', ent_field = 'ents', lang = tgt_lang)

with open(json_path.replace('.json', '_fa.json'), 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii = False)

for line in data:
    text_en_tokenized.append(line['ingredients_en'])
    text_it_tokenized.append(line['ingredients_it'])

parallel_list = []

for en, it in zip(text_en_tokenized, text_it_tokenized):
    parallel_list.append(f'{en} ||| {it}')

output_filename = json_path.replace('.json', '_fa.txt')
with open(output_filename, 'w', encoding='utf8') as f:
    for line in parallel_list:
        f.write(line + '\n')