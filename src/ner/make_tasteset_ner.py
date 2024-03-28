from datasets import Dataset
import os
import json
import pandas as pd
# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-PE_en-it_spaced.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_LOC_en-it_spaced.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_en-it_spaced.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_TS_mdeberta_xlwa_en-it_EW-TT-PE_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mdeberta_align_E4_DEV94.0_20240326-10-49-49_ls.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_TS_mdeberta_xlwa_en-it_EW-TT-MT_LOC_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mdeberta_align_E3_DEV95.0_2024-03-26-14-21-06_ls.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_aligned_spaced_TS_mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_U0_S1_ING_P0.5_DROP1_mdeberta_align_E8_DEV98.0_2024-03-27-10-14-18_ls.json'
json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_en-it_context.json'

with open(json_path, 'r', encoding='utf8') as f:
    dataset = json.load(f)

import sys
sys.path.append('/home/pgajo/food/src')

from utils import tasteset_to_label_studio

dataset = tasteset_to_label_studio(dataset['annotations'])
print(dataset)
df = pd.DataFrame(data=dataset)
dataset = Dataset.from_pandas(df)
print(dataset)
datasetdict = dataset.train_test_split(0.1)
dataset_name = json_path.split('/')[-1].split('.json')[0]
repo_name = f'pgajo/{dataset_name}'
print('repo name:', repo_name)
dataset_dir = '/home/pgajo/food/datasets/ner'
out_path = os.path.join(dataset_dir, dataset_name + '_ner')
datasetdict.save_to_disk(out_path)
print(out_path)