import os
import pandas as pd
from wordalignutils import WADataset
from transformers import AutoTokenizer
import torch

data_path_train = '/home/pgajo/working/food/data/XL-WA/data/.ready/it'
dataset = WADataset.load_from_disk(data_path_train)


train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size = 32, shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size = 32, shuffle = True)

results_path = f'/home/pgajo/working/food/src/word_alignment/XL-WA/results/{dataset.lang_id}'
if not os.path.isdir(results_path):
    os.mkdir(results_path)

