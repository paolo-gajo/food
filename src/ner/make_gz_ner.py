from datasets import Dataset, DatasetDict
import os
import json
import pandas as pd

json_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced.json'

with open(json_path, 'r', encoding='utf8') as f:
    dataset_raw = json.load(f)

df = pd.DataFrame(data=dataset_raw)
dataset = Dataset.from_pandas(df[['data', 'annotations']])
print(dataset)
datasetdict = DatasetDict()
datasetdict['train'] = dataset
print(datasetdict['train'][0])
dataset_name = json_path.split('/')[-1].split('.')[0] + '_ner_test'
repo_name = f'pgajo/{dataset_name}'
print('repo name:', repo_name)
dataset_dir = '/home/pgajo/food/datasets/ner'
datasetdict.save_to_disk(os.path.join(dataset_dir, dataset_name))