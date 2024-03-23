import pandas as pd
from datasets import load_from_disk
from evaluate import evaluator
from ner_utils import make_ner_sample, get_ner_classes
from transformers import AutoTokenizer, AutoModelForTokenClassification
from icecream import ic
model_name = "bert-base-multilingual-cased"
# model_name = "bert-base-uncased"
# model_name = "bert-large-uncased"
# model_name = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

data_name_test = '/home/pgajo/food/datasets/GZ-GOLD-NER-ALIGN_105_testonly'
dataset_test = load_from_disk(data_name_test)
label_field = 'annotations'
# print(dataset_test['train'][label_field])
label_list, label2id, id2label = get_ner_classes(dataset_test, label_field=label_field)

tokenized_dataset_test = dataset_test.map(lambda example: make_ner_sample(example, tokenizer, label2id, text_name='ingredients', label_field=label_field))
tokenized_dataset_test = tokenized_dataset_test.remove_columns(['data', 'annotations'])

