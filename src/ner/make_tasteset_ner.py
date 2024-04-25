from datasets import Dataset
import os
import json
import pandas as pd
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import TASTEset


# json_path = '/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_multi_ctx.json'

# en-it
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it_preds_giza.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it_preds_fast-align.json'

# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mbert_xlwa_en-it_EW-TT-MT_multi_ctx_P0.1_en-it_ME3_2024-04-22-11-55-05_TEST46.44_ls.json' old
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mdeberta_xlwa_en-it_EW-TT-MT_multi_ctx_P0.1_en-it_ME3_2024-04-22-16-29-38_TEST60.59_ls.json' old

# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mbert_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.1_en-it-es_ME3_2024-04-24-00-24-38_TEST_GZ=49.46_ls.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mdeberta_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-24-06-08-49_TEST_GZ=61.87_ls.json' # combine

# en-es
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es_preds_giza.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es_preds_fast-align.json'

# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mbert_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-24-03-17-41_TEST_MCR=70.61_ls.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_aligned_mdeberta-v3-base_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-23-21-21-18_TEST_MCR=75.56_ls.json' # combine

# en-it-es
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it-es/SW-TASTE_DEEPL_aligned_en-it-es_giza.json'
# json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it-es/SW-TASTE_DEEPL_aligned_en-it-es_fast-align.json'
json_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it-es/SW-TASTE_DEEPL_aligned_en-it-es_mDeBERTa.json'

with open(json_path, 'r', encoding='utf8') as f:
    dataset = json.load(f)

languages = set([key[-2:] for key in dataset[0]['data'].keys()])
print(languages)

# del_list = [f'ents_{lang}' for lang in languages]
# dataset = TASTEset.tasteset_to_label_studio(dataset, languages=languages, text_field='ingredients', del_list=del_list)

df = pd.DataFrame(data=dataset)
dataset = Dataset.from_pandas(df)

datasetdict = dataset.train_test_split(0.1)
dataset_name = json_path.split('/')[-1].split('.json')[0]

dataset_dir = '/home/pgajo/food/datasets/ner'
out_path = os.path.join(dataset_dir, dataset_name + '_ner')
datasetdict.save_to_disk(out_path)
print('out_path:', out_path)