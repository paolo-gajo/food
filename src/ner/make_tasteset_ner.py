from datasets import Dataset
import os
import json
import pandas as pd

# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_en-it_context_fix_TS.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_multi_context_TS.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-20-36-43_TEST61.0.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_TEST49.5_ls.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-04-02-17-37-26_TEST53.0.json'
json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-04-03-06-12-17_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-04-03-10-55-06_TEST63.json'

with open(json_path, 'r', encoding='utf8') as f:
    dataset = json.load(f)
languages = set([key[-2:] for key in dataset['annotations'][0].keys()])
print(languages)
import sys
sys.path.append('/home/pgajo/food/src')

from utils import tasteset_to_label_studio

dataset = tasteset_to_label_studio(dataset['annotations'], languages=languages)
# print(dataset)
df = pd.DataFrame(data=dataset)
dataset = Dataset.from_pandas(df)
# print(dataset)
datasetdict = dataset.train_test_split(0.1)
dataset_name = json_path.split('/')[-1].split('.json')[0]
repo_name = f'pgajo/{dataset_name}'
print('repo name:', repo_name)
dataset_dir = '/home/pgajo/food/datasets/ner'
out_path = os.path.join(dataset_dir, dataset_name + '_ner')
datasetdict.save_to_disk(out_path)
print(out_path)