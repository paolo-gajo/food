import os
import pandas as pd
from wordalignutils import XLWADataset
from transformers import AutoTokenizer

# pd.options.display.max_colwidth = 100

lang_list = [
    # 'ru',
    # 'nl',
    'it',
    # 'pt',
    # 'et',
    # 'es',
    # 'hu',
    # 'da',
    # 'bg',
    # 'sl',
]

data_path = '/home/pgajo/working/food/data/XL-WA/data'
dataset = XLWADataset(lang_list,
                        data_path,
                        # 20
                        )
print(dataset.__class__)
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

dataset = dataset.format_and_tokenize(tokenizer)
print(dataset.__class__)
data_train_path = os.path.join(data_path, f'.ready/{dataset.lang_id}')
if not os.path.isdir(data_train_path):
    os.mkdir(data_train_path)
    
dataset.save_to_disk(data_train_path)


