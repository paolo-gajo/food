from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from icecream import ic
import os
import re
import pandas as pd
import sys
sys.path.append('/home/pgajo/food/src')
from utils import data_loader
import torch
from tqdm.auto import tqdm
import evaluate

login(os.environ['HF_TOKEN'])
from ner_utils import make_ner_sample, get_ner_classes
# import sys
# # sys.path.append('/home/pgajo/food/src')
# from utils import make_ner_sample

data_name = '/home/pgajo/food/datasets/EW-TT-PE_en-it_spaced'
data_name_simple = data_name.split('/')[-1]
dataset = load_from_disk(data_name)
# dataset['train'] = dataset['train'].select(range(2))
# dataset['test'] = dataset['test'].select(range(2))
# dataset.save_to_disk(os.path.join('/home/pgajo/food/datasets', data_name.split('/')[-1]))
print(dataset)

raw_labels, label_list, label2id, id2label = get_ner_classes(dataset)
ic(raw_labels)
ic(label_list)

from transformers import AutoTokenizer


model_name = "bert-base-multilingual-cased"
# model_name = "bert-base-uncased"=======
model_name = "bert-large-uncased"
# model_name = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# for i, example in enumerate(dataset['train']):
#     output = make_ner_sample(dataset['train'][i], tokenizer, label2id, lang='it')
#     if not len(output['input_ids']) == len(output['labels']):
#         print(output)
#         print(len(output['input_ids']))
#         print(len(output['labels']))
#         for i, (id, label) in enumerate(zip([tokenizer.decode(token) for token in output['input_ids']], output['labels'])):
#             print(i, id, label, sep = '\t')

languages_traindev = [
    'en',
    'it'
    ]
languages_test = [
    'en',
    'it'
    ]
df_train = pd.DataFrame()
df_dev = pd.DataFrame()

for lang in languages_traindev:
    df_train = pd.concat([df_train, pd.DataFrame([make_ner_sample(sample, tokenizer, label2id, lang, label_list=raw_labels) for sample in dataset['train']])])
    df_dev = pd.concat([df_dev, pd.DataFrame([make_ner_sample(sample, tokenizer, label2id, lang, label_list=raw_labels) for sample in dataset['test']])])

dataset_train = Dataset.from_pandas(df_train)
dataset_dev = Dataset.from_pandas(df_dev)

tokenized_dataset = DatasetDict()
tokenized_dataset['train'] = dataset_train
tokenized_dataset['dev'] = dataset_dev

# tokenized_dataset = dataset.map(lambda example: make_ner_sample(example, tokenizer, label2id),
                                                                # batched=True,
                                                                # batch_size=1,
                                                                # )
# tokenized_dataset.set_format('torch')
# tokenized_dataset = tokenized_dataset.remove_columns(['data', 'predictions'])
print(tokenized_dataset)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Get the NER labels first, and then create a function that passes your true predictions
# and true labels to [compute](https://huggingface.co/docs/evaluate/main/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the scores:

import numpy as np

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model_name_simple = model_name.split('/')[-1]
model = AutoModelForTokenClassification.from_pretrained(
                                                    model_name,
                                                    num_labels=len(label_list),
                                                    id2label=id2label,
                                                    label2id=label2id
                                                )

model_dir = os.path.join('/home/pgajo/food/models', model_name_simple)

training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    # load_best_model_at_end=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

data_name_test = '/home/pgajo/food/datasets/GZ-GOLD-NER-ALIGN_105_spaced_testonly'
data_name_test_simple = data_name_test.split('/')[-1]
dataset_test_raw = load_from_disk(data_name_test)
label_field = 'annotations'
# print(dataset_test_raw['train'][label_field])
# label_list, label2id, id2label = get_ner_classes(dataset_test_raw, label_field=label_field)
# ic(label_list)

df_test = pd.DataFrame()

verbose = 0
for lang in languages_test:
    sample_buffer = []
    for i, sample in enumerate(dataset_test_raw['train']):
        ner_sample = make_ner_sample(sample,
                                    tokenizer,
                                    label2id,
                                    text_name='text',
                                    label_field=label_field,
                                    lang=lang,
                                    label_list=raw_labels)
        # if len(ner_sample['labels']) > 512:
        if verbose:
            # print('text', tokenizer.decode(ner_sample['input_ids']))
            # print('input_ids', ner_sample['input_ids'])
            # print('decoded ids', [tokenizer.decode(id) for id in ner_sample['input_ids']])
            # print('labels', ner_sample['labels'])
            print('line number:', i)
            for decoded_token, label in zip([tokenizer.decode(id) for id in ner_sample['input_ids']], ner_sample['labels']):
                if label != -100:
                    print(decoded_token, (id2label[label] if label in id2label.keys() else 'None'), sep = '\t')
            print('--------------------------------------')
            # raise Exception("Length > 512")
        sample_buffer.append(ner_sample)
    df_test = pd.concat([df_test, pd.DataFrame(sample_buffer)])

dataset_test = Dataset.from_pandas(df_test)

tokenized_dataset_test = DatasetDict()
tokenized_dataset_test['test'] = dataset_test
tokenized_dataset_test.set_format('torch')
ic(tokenized_dataset_test)

batch_size = 16
dataset_test = data_loader(tokenized_dataset_test, 
                    batch_size,
                    # n_rows=100
                    )

epoch = 0
# eval on test
epoch_test_loss = 0
model.eval()
split = 'test'
progbar = tqdm(enumerate(dataset_test[split]),
                        total=len(dataset_test[split]),
                        desc=f"{split} - epoch {epoch + 1}")

columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']

all_preds = []
all_trues = []
texts = []

def compute_metrics(predictions, labels):
    # predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }
    return results


for i, batch in progbar:
    with torch.inference_mode():
        input = {k: batch[k].to('cuda') for k in columns}
        outputs = model(**input)
        preds = torch.argmax(outputs.logits, dim = 2)
        all_preds.extend(preds.tolist())
        all_trues.extend(batch['labels'].tolist())
        texts.extend([tokenizer.decode(ids) for ids in batch['input_ids']])

epoch_test_loss /= len(dataset[split])
print(compute_metrics(all_preds, all_trues))
print(pd.DataFrame(compute_metrics(all_preds, all_trues)))

result_dict = {
    'text': texts,
    'gold': all_trues,
    'predictions': all_preds,
}

result_df = pd.DataFrame(result_dict)
csv_dir = f'/home/pgajo/food/results/ner/{data_name_test_simple}'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
result_df.to_csv(os.path.join(csv_dir, f'{model_name_simple}_predictions.csv'))