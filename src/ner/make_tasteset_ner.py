from datasets import Dataset
import os
import json
import pandas as pd
# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-PE_en-it_spaced.json'
json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_en-it_spaced.json'

with open(json_path, 'r', encoding='utf8') as f:
    dataset = json.load(f)

import sys
sys.path.append('/home/pgajo/food/src')

# from utils import tasteset_to_label_studio

# dataset = tasteset_to_label_studio(dataset['annotations'])
# print(dataset)
df = pd.DataFrame(data=dataset)
dataset = Dataset.from_pandas(df)
print(dataset)
datasetdict = dataset.train_test_split(0.1)
dataset_name = json_path.split('/')[-1].split('.')[0]
repo_name = f'pgajo/{dataset_name}'
print('repo name:', repo_name)
dataset_dir = '/home/pgajo/food/datasets'
datasetdict.save_to_disk(os.path.join(dataset_dir, dataset_name + '_ner'))