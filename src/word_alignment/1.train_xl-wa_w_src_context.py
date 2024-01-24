import os
import pandas as pd
from typing import List, Union
from wlutils import XLWADataset

class XLWADataset:

    def __init__(self, languages: List, data_path: str, n_rows: Union[int, None] = None) -> List[dict]:
        
        self.train, self.dev, self.test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.lang_id = '-'.join(languages)
        
        for lang in languages:

            df_train_tmp = pd.read_csv(os.path.join(data_path, f'{lang}/train.tsv'), sep='\t', header=None)
            df_train_tmp.columns = ['src', 'tgt', 'alignments']
            df_train_tmp['lang'] = lang
            df_train_tmp['split'] = 'train'

            df_dev_tmp = pd.read_csv(os.path.join(data_path, f'{lang}/dev.tsv'), sep='\t', header=None)
            df_dev_tmp.columns = ['src', 'tgt', 'alignments']
            df_dev_tmp['lang'] = lang
            df_dev_tmp['split'] = 'dev'

            df_test_tmp = pd.read_csv(os.path.join(data_path, f'{lang}/test.tsv'), sep='\t', header=None)
            df_test_tmp.columns = ['src', 'tgt', 'alignments']
            df_test_tmp['lang'] = lang
            df_test_tmp['split'] = 'test'

            # concat train and dev
            self.train = pd.concat([self.train, df_train_tmp])
            self.dev = pd.concat([self.dev, df_dev_tmp])
            self.test = pd.concat([self.test, df_test_tmp])

            self.train = self.train[:n_rows]
            self.dev = self.dev[:n_rows]
            self.test = self.test[:n_rows]

            # Adding a new column for span alignments
            self.train['span_alignments'] = self.train.apply(lambda row: self.convert_alignments(row[0], row[1], row[2]), axis=1)
            self.dev['span_alignments'] = self.dev.apply(lambda row: self.convert_alignments(row[0], row[1], row[2]), axis=1)
            self.test['span_alignments'] = self.test.apply(lambda row: self.convert_alignments(row[0], row[1], row[2]), axis=1)

            self.dict = {
                'train': df_train,
                'dev': df_dev,
                'test': df_test,
            }
    
    @staticmethod
    def calculate_spans(sentence):
        spans = []
        start = 0
        for word in sentence.split():
            end = start + len(word)
            spans.append((start, end))
            start = end + 1  # Add 1 for the space character
        return spans

    def convert_alignments(self, src_sentence, tgt_sentence, alignments):
        src_spans = self.calculate_spans(src_sentence)
        tgt_spans = self.calculate_spans(tgt_sentence)

        converted_alignments = []
        for alignment in alignments.split():
            src_idx, tgt_idx = map(int, alignment.split('-'))
            src_span = src_spans[src_idx]
            tgt_span = tgt_spans[tgt_idx]
            converted_alignments.append(((src_span[0],src_span[1]),(tgt_span[0], tgt_span[1])))

        return converted_alignments


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

data_path = '/home/pgajo/working/food/src/word_alignment/XL-WA/data/'
dataset = XLWADataset(lang_list,
                        data_path,
                        20
                        )

for lang in lang_list:

    df_train_tmp = pd.read_csv(f'/home/pgajo/working/food/src/word_alignment/XL-WA/data/{lang}/train.tsv', sep='\t', header=None)
    df_train_tmp.columns = ['src', 'tgt', 'alignments']
    df_train_tmp['lang'] = lang
    df_train_tmp['split'] = 'train'

    df_dev_tmp = pd.read_csv(f'/home/pgajo/working/food/src/word_alignment/XL-WA/data/{lang}/dev.tsv', sep='\t', header=None)
    df_dev_tmp.columns = ['src', 'tgt', 'alignments']
    df_dev_tmp['lang'] = lang
    df_dev_tmp['split'] = 'dev'

    df_test_tmp = pd.read_csv(f'/home/pgajo/working/food/src/word_alignment/XL-WA/data/{lang}/test.tsv', sep='\t', header=None)
    df_test_tmp.columns = ['src', 'tgt', 'alignments']
    df_test_tmp['lang'] = lang
    df_test_tmp['split'] = 'test'

    # concat train and dev
    df_train = pd.concat([df_train, df_train_tmp])
    df_dev = pd.concat([df_dev, df_dev_tmp])
    df_test = pd.concat([df_test, df_test_tmp])

from transformers import AutoTokenizer
model_name = 'bert-base-multilingual-cased'
# model_name = 'microsoft/mdeberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare data

import torch
torch.set_printoptions(linewidth=10000)

df_dict_formatted = {
    'train': [],
    'dev': [],
    'test': [],
}

max_len = 0

# create a word_src_j -> sentence_tgt_i dataset
for key in df_dict.keys():
    tgt_sentences = df_dict[key]['tgt'].tolist()
    src_sentences = df_dict[key]['src'].tolist()
    src_split_sentences = [sentence.split() for sentence in df_dict[key]['src'].tolist()]
    
    for i, sentence_src in enumerate(src_split_sentences):
        alignments = df_dict[key]['span_alignments'].to_list()[i]
        # print('sentence alignments', alignments)
        for j, alignment in enumerate(alignments):
            entry = {}
            # print('word alignment', alignment)
            src_start = alignment[0][0]
            # print('src_start', src_start)
            src_end = alignment[0][1]
            # print('src_end', src_end)
            tgt_start = alignment[1][0]
            # print('tgt_start', tgt_start)
            tgt_end = alignment[1][1]
            # print('tgt_end', tgt_end)
            entry['id_sentence'] = i
            entry['id_alignment'] = j
            entry['query'] = src_sentences[i][:src_start] +\
                '• ' + src_sentences[i][src_start:src_end] +\
                ' •' + src_sentences[i][src_end:]
            # print(entry['query'])
            entry['context'] = tgt_sentences[i]
            entry['answer'] = tgt_sentences[i][tgt_start:tgt_end]
            entry['answer_start'] = tgt_start
            entry['answer_end'] = tgt_end
            char_check = entry['context'][entry['answer_start']:entry['answer_end']]
            query_encoding = tokenizer(entry['query'])
            context_encoding = tokenizer(entry['context'])
            entry['answer_start_token'] = context_encoding.char_to_token(entry['answer_start']) + len(query_encoding['input_ids']) - 1
            entry['answer_end_token'] = context_encoding.char_to_token(entry['answer_end']-1) + len(query_encoding['input_ids'])
            
            input_encoding = tokenizer(entry['query'],
                                       entry['context']
                                    )
            
            if len(input_encoding['input_ids']) > max_len:
                max_len = len(input_encoding['input_ids'])
            
            token_check = tokenizer.decode(input_encoding['input_ids'][entry['answer_start_token']:entry['answer_end_token']])
            if not entry['query']:
                print('query missing')
            
            if not char_check == ''.join((token_check).split()):
                print('----------------------------------------------')
                print(entry['id_sentence'], entry['id_alignment'])
                print('src_sentences[i]', src_sentences[i])
                print('query_start', src_start)
                print('query_end', src_end)
                print("entry['query']", entry['query'])
                print("entry['context']", entry['context'])
                print("entry['answer']", entry['answer'])
                print("entry['answer_start']", entry['answer_start'])
                print("entry['answer_end']", entry['answer_end'])
                print('########### char_check', char_check)
                print('query_encoding', query_encoding)
                print('context_encoding', context_encoding)
                print(entry['answer_start_token'])
                print(entry['answer_end_token'])
                print('########### token_check', token_check)
                print('########### ''.join((token_check).split())', ''.join((token_check).split()))
                print('###########', char_check == ''.join((token_check).split()))

            # test if the answer_start_token:answer_end_token is the same as the answer
            # print(tokenizer.decode(tokenizer(entry['context'])['input_ids'][entry['answer_start_token']:entry['answer_end_token']]))
            
            # print(entry)
            df_dict_formatted[key].append(entry)

print(f'max_len', max_len)

# convert to Dataset format

from datasets import Dataset, DatasetDict
dataset_train = Dataset.from_list(df_dict_formatted['train'])
dataset_dev = Dataset.from_list(df_dict_formatted['dev'])
dataset_test = Dataset.from_list(df_dict_formatted['test'])

dataset = DatasetDict({
    'train': dataset_train,
    'validation': dataset_dev,
    'test': dataset_test,
    })

# print(dataset)
def tokenize_function(example):
    return tokenizer(example['query'], example['context'], 
                     padding='max_length',
                     max_length=max_len,
                     )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'answer_start_token', 'answer_end_token'])
print(tokenized_dataset)


# make data loaders

train_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size = 32, shuffle = True)
val_loader = torch.utils.data.DataLoader(tokenized_dataset['validation'], batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size = 32, shuffle = True)

# prep training setup

import os
results_path = f'/home/pgajo/working/food/src/word_alignment/XL-WA/results/{lang_id}'
if not os.path.isdir(results_path):
    os.mkdir(results_path)

# Lists to store metrics
train_losses, train_f1s, train_exact_matches, train_f1s_squad_evaluate, train_exact_matches_squad_evaluate, train_f1s_squad_datasets, train_exact_matches_squad_datasets = [], [], [], [], [], [], []
val_losses, val_f1s, val_exact_matches, val_f1s_squad_evaluate, val_exact_matches_squad_evaluate, val_f1s_squad_datasets, val_exact_matches_squad_datasets = [], [], [], [], [], [], []
test_losses, test_f1s, test_exact_matches, test_f1s_squad_evaluate, test_exact_matches_squad_evaluate, test_f1s_squad_datasets, test_exact_matches_squad_datasets = [], [], [], [], [], [], []

avg_type = 'micro'

from transformers import AutoModelForQuestionAnswering
import torch
torch.set_printoptions(linewidth=1000)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
model = torch.nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# Training setup
from tqdm.auto import tqdm
import time
current_timeanddate = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import uuid
from evaluate import load
squad_metric_evaluate = load("squad_v2")
from datasets import load_metric
squad_metric_datasets = load_metric("squad")

# training loop

# Initialize variables to track the best model and early stopping
best_results_val_squad = 0.0
best_model = None
early_stopping_counter = 0
early_stopping_patience = 2  # Set your patience for early stopping

# Initialize DataFrame for storing metrics
df = pd.DataFrame()

epochs = 10
whole_train_eval_time = time.time()

print_every = 100

for epoch in range(epochs):
    epoch_time = time.time()

    ################################################### Training Phase
    model.train()
    epoch_train_loss = 0
    
    # Initialize tqdm progress bar
    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

    train_all_preds, train_all_true = [], []
    train_qa_preds_evaluate, train_qa_trues_evaluate = [], []
    train_qa_preds_datasets, train_qa_trues_datasets = [], []

    for batch_idx, batch in train_progress_bar: 
        optimizer.zero_grad()
        inputs = {
            "input_ids": batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            "attention_mask": batch['attention_mask'].to(device),
            "start_positions": batch['answer_start_token'].to(device),
            "end_positions": batch['answer_end_token'].to(device),
        }

        outputs = model(**inputs)
        loss = outputs[0].mean()
        epoch_train_loss += loss.item()
        
        # Update tqdm postfix to display loss
        loss_tmp = round(epoch_train_loss / (batch_idx + 1), 4)
        train_progress_bar.set_postfix({'Loss': loss_tmp})

        loss.backward()
        optimizer.step()

        # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
        for i in range(len(outputs['start_logits'])):
            outputs['start_logits'][i] = torch.where(inputs['token_type_ids'][i]!=0, outputs['start_logits'][i], inputs['token_type_ids'][i]-10000)
            outputs['end_logits'][i] = torch.where(inputs['token_type_ids'][i]!=0, outputs['end_logits'][i], inputs['token_type_ids'][i]-10000)

        start_preds = torch.argmax(outputs['start_logits'], dim=1)
        end_preds = torch.argmax(outputs['end_logits'], dim=1)

        # inspect predictions and answers
        # for i in range(len(start_preds)):
        #     if loss_tmp < 1:
        #         print(tokenizer.decode(inputs['input_ids'][i][start_preds[i]:end_preds[i]]))
        #         print(tokenizer.decode(inputs['input_ids'][i][inputs['start_positions'][i]:inputs['end_positions'][i]]))
        #         print('-----------')
            
        pred_batch = [el for el in zip(start_preds.tolist(), end_preds.tolist())]
        true_batch = [el for el in zip(inputs["start_positions"].tolist(), inputs["end_positions"].tolist())]
        
        train_all_preds.extend(pred_batch)
        train_all_true.extend(true_batch)

        pred_batch_ids = [str(uuid.uuid4()) for i in range(len(start_preds))]

        for i, pair in enumerate(pred_batch):
            if pair[0] >= pair[1]:
                text_pred = ''
            else:
                text_pred = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
                if not isinstance(text_pred, str):
                    text_pred = ''
            
            entry_evaluate = {
                'prediction_text': text_pred,
                'id': pred_batch_ids[i],
                'no_answer_probability': 0
            }

            entry_datasets = {
                'prediction_text': text_pred,
                'id': pred_batch_ids[i],
                # 'no_answer_probability': 0
            }

            train_qa_preds_evaluate.append(entry_evaluate)
            train_qa_preds_datasets.append(entry_datasets)
        
        for i, pair in enumerate(true_batch):
            text_true = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
            entry = {
                'answers': {
                    'answer_start': [true_batch[0][0]],
                    'text': [text_true],
                    },
                    'id': pred_batch_ids[i]
                }
            
            train_qa_trues_evaluate.append(entry)
            train_qa_trues_datasets.append(entry)


    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)   

    # Calculate training metrics
    train_pred_flat = [p for pair in train_all_preds for p in pair]
    train_true_flat = [t for pair in train_all_true for t in pair]
    train_f1 = f1_score(train_true_flat, train_pred_flat, average=avg_type)
    train_exact_match = accuracy_score(train_true_flat, train_pred_flat)
    train_f1s.append(train_f1)
    train_exact_matches.append(train_exact_match)

    results_train_squad_evaluate = squad_metric_evaluate.compute(predictions=train_qa_preds_evaluate, references=train_qa_trues_evaluate)
    results_train_squad_datasets = squad_metric_datasets.compute(predictions=train_qa_preds_datasets, references=train_qa_trues_datasets)
    # print(results_train_squad_evaluate)
    # print(results_train_squad_datasets)
    train_f1s_squad_evaluate.append(results_train_squad_evaluate['f1'])
    train_exact_matches_squad_evaluate.append(results_train_squad_evaluate['exact'])
    train_f1s_squad_datasets.append(results_train_squad_datasets['f1'])
    train_exact_matches_squad_datasets.append(results_train_squad_datasets['exact_match'])

    ################################################### Validation Phase
    model.eval()
    epoch_val_loss = 0
    val_all_preds, val_all_true = [], []
    val_qa_preds_evaluate, val_qa_trues_evaluate = [], []
    val_qa_preds_datasets, val_qa_trues_datasets = [], []

    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)): 
        with torch.inference_mode():
            inputs = {
                "input_ids": batch['input_ids'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                "attention_mask": batch['attention_mask'].to(device),
                "start_positions": batch['answer_start_token'].to(device),
                "end_positions": batch['answer_end_token'].to(device),
            }
            
            outputs = model(**inputs)
            loss = outputs[0].mean()
            epoch_val_loss += loss.item()

            # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
            for i in range(len(outputs['start_logits'])):
                outputs['start_logits'][i] = torch.where(inputs['token_type_ids'][i]!=0, outputs['start_logits'][i], inputs['token_type_ids'][i]-10000)
                outputs['end_logits'][i] = torch.where(inputs['token_type_ids'][i]!=0, outputs['end_logits'][i], inputs['token_type_ids'][i]-10000)

        start_preds = torch.argmax(outputs['start_logits'], dim=1)
        end_preds = torch.argmax(outputs['end_logits'], dim=1)

        pred_batch = [el for el in zip(start_preds.tolist(), end_preds.tolist())]
        true_batch = [el for el in zip(inputs["start_positions"].tolist(), inputs["end_positions"].tolist())]

        val_all_preds.extend(pred_batch)
        val_all_true.extend(true_batch)

        pred_batch_ids = [str(uuid.uuid4()) for i in range(len(start_preds))]

        for i, pair in enumerate(pred_batch):
            if pair[0] >= pair[1]:
                text_pred = ''
            else:
                text_pred = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
                if not isinstance(text_pred, str):
                    text_pred = ''
            
            entry_evaluate = {
                'prediction_text': text_pred,
                'id': pred_batch_ids[i],
                'no_answer_probability': 0
            }

            entry_datasets = {
                'prediction_text': text_pred,
                'id': pred_batch_ids[i],
                # 'no_answer_probability': 0
            }

            val_qa_preds_evaluate.append(entry_evaluate)
            val_qa_preds_datasets.append(entry_datasets)
        
        for i, pair in enumerate(true_batch):
            text_true = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
            entry = {
                'answers': {
                    'answer_start': [true_batch[0][0]],
                    'text': [text_true],
                    },
                    'id': pred_batch_ids[i]
                }
            
            val_qa_trues_evaluate.append(entry)
            val_qa_trues_datasets.append(entry)


    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)   

    # Calculate training metrics
    val_pred_flat = [p for pair in val_all_preds for p in pair]
    val_true_flat = [t for pair in val_all_true for t in pair]
    val_f1 = f1_score(val_true_flat, val_pred_flat, average=avg_type)
    val_exact_match = accuracy_score(val_true_flat, val_pred_flat)
    val_f1s.append(val_f1)
    val_exact_matches.append(val_exact_match)

    results_val_squad_evaluate = squad_metric_evaluate.compute(predictions=val_qa_preds_evaluate, references=val_qa_trues_evaluate)
    results_val_squad_datasets = squad_metric_datasets.compute(predictions=val_qa_preds_datasets, references=val_qa_trues_datasets)
    # print(results_val_squad_evaluate)
    # print(results_val_squad_datasets)
    val_f1s_squad_evaluate.append(results_val_squad_evaluate['f1'])
    val_exact_matches_squad_evaluate.append(results_val_squad_evaluate['exact'])
    val_f1s_squad_datasets.append(results_val_squad_datasets['f1'])
    val_exact_matches_squad_datasets.append(results_val_squad_datasets['exact_match'])

    # Log Epoch Metrics
    print("\n-------Epoch ", epoch + 1, 
          "-------"
          "\nTraining Loss:", train_losses[-1],
        #   f"\nTraining F1 {avg_type}:", train_f1s[-1],
        #   "\nTraining Exact Match:", train_exact_matches[-1],
          "\nTraining F1 Squad Evaluate:", results_train_squad_evaluate['f1'],
          "\nTraining Exact Squad Evaluate:", results_train_squad_evaluate['exact'],
        #   "\nTraining F1 Squad Datasets:", results_train_squad_datasets['f1'],
        #   "\nTraining Exact Squad Datasets:", results_train_squad_datasets['exact_match'],
        #   "\nValidation Loss:", val_losses[-1],
        #   f"\nValidation F1 {avg_type}:", val_f1s[-1],
        #   "\nValidation Exact Match:", val_exact_matches[-1],
          "\nValidation F1 Squad Evaluate:", results_val_squad_evaluate['f1'],
          "\nValidation Exact Squad Evaluate:", results_val_squad_evaluate['exact'],
        #   "\nValidation F1 Squad Datasets:", results_val_squad_datasets['f1'],
        #   "\nValidation Exact Squad Datasets:", results_val_squad_datasets['exact_match'],
          "\nTime: ", (time.time() - epoch_time),
          "\n-----------------------",
          "\n\n")
    
    test_losses.append('')
    test_f1s.append('')
    test_exact_matches.append('')
    test_f1s_squad_evaluate.append('')
    test_exact_matches_squad_evaluate.append('')
    test_f1s_squad_datasets.append('')
    test_exact_matches_squad_datasets.append('')

    # Save metrics to DataFrame and CSV
    df = pd.DataFrame({
        'epoch': range(epoch+1), 
        'train_loss': train_losses, 
        # f'train_f1_{avg_type}': train_f1s,
        # 'train_exact_match': train_exact_matches,
        'train_f1s_squad_evaluate': train_f1s_squad_evaluate,
        'train_exact_matches_squad_evaluate': train_exact_matches_squad_evaluate,
        # 'train_f1s_squad_datasets': train_f1s_squad_datasets,
        # 'train_exact_matches_squad_datasets': train_exact_matches_squad_datasets,
        # 'val_loss': val_losses, 
        # f'val_f1_{avg_type}': val_f1s, 
        # 'val_exact_match': val_exact_matches,
        'val_f1s_squad_evaluate': val_f1s_squad_evaluate,
        'val_exact_matches_squad_evaluate': val_exact_matches_squad_evaluate,
        # 'val_f1s_squad_datasets': val_f1s_squad_datasets,
        # 'val_exact_matches_squad_datasets': val_exact_matches_squad_datasets,
        # 'test_loss': test_losses, 
        # f'test_f1_{avg_type}': test_f1s, 
        # 'test_exact_match': test_exact_matches,
        'test_f1s_squad_evaluate': test_f1s_squad_evaluate,
        'test_exact_matches_squad_evaluate': test_exact_matches_squad_evaluate,
        # 'test_f1s_squad_datasets': test_f1s_squad_datasets,
        # 'test_exact_matches_squad_datasets': test_exact_matches_squad_datasets,
    })
    csv_filename = f"{current_timeanddate}_{model_name.split('/')[-1]}_metrics.csv"
    df.to_csv(os.path.join(results_path, csv_filename), index=False)

    # Check for Best Model and Implement Early Stopping
    if results_val_squad_datasets['exact_match'] > best_results_val_squad or results_val_squad_evaluate['exact'] > best_results_val_squad:
        best_results_val_squad = max(results_val_squad_datasets['exact_match'], results_val_squad_evaluate['exact'])
        del best_model
        best_model = model
        epochs_best = epoch + 1
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Total training and evaluation time: ", (time.time() - whole_train_eval_time))

# save best model

from huggingface_hub import login
token='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'
login(token=token)
model_save_name = f"pgajo/{model_name.split('/')[-1]}-xl-wa-{lang_id}-{epochs_best}-epochs"
model.module.push_to_hub(model_save_name)

from huggingface_hub import whoami, create_repo, ModelCard, ModelCardData

user = whoami(token=token)
repo_id = model_save_name
url = create_repo(repo_id, exist_ok=True)
card_data = ModelCardData(language='en', license='mit', library_name='pytorch')
card = ModelCard.from_template(
    card_data,
    model_id=model_name.split('/')[-1],
    model_description=f"Word alignment model fine-tuned on XL-WA ({lang_id}) for {epochs_best} epochs.",
    developers="Paolo Gajo",
    repo="https://github.com/paolo-gajo/food",
)
card.push_to_hub(repo_id)

# load best model for testing evaluation

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
torch.set_printoptions(linewidth=1000)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

try:
    model
except NameError:
    model_exists = False
else:
    model_exists = True

if model_exists:
    pass
else:
    # load previously saved model
    # model_name = "pgajo/bert-base-multilingual-cased-xl-wa-it-es-9-epochs"
    # model_name = 'pgajo/mdeberta-v3-base-xl-wa-en-it-10-epochs'
    # model_name = 'pgajo/bert-base-multilingual-cased-xl-wa-it-10-epochs'
    # model_name = 'pgajo/bert-base-multilingual-cased-xl-wa-es-9-epochs'
    model_name = 'pgajo/bert-base-multilingual-cased-xl-wa-it-2-epochs'
    best_model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    best_model = torch.nn.DataParallel(best_model)
    # model_name = 'bert-base-multilingual-cased'
    # model_name = 'microsoft/mdeberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Training setup
from tqdm.auto import tqdm
import time
current_timeanddate = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import uuid
from evaluate import load
squad_metric_evaluate = load("squad_v2")
from datasets import load_metric
squad_metric_datasets = load_metric("squad")

# evaluate model on test set

try:
    df
except NameError:
    df_exists = False
else:
    df_exists = True

if df_exists:
    pass
else:
    csv_filename = "/home/pgajo/working/food/src/word_alignment/XL-WA/results/it/20240117-125659_bert-base-multilingual-cased_metrics.csv"
    df = pd.read_csv(csv_filename)

# Testing Phase
best_model.eval()
epoch_test_loss = 0
test_all_preds, test_all_true = [], []
test_qa_preds, test_qa_trues = [], []
for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)): 
    with torch.inference_mode():
        inputs = {
            "input_ids": batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            "attention_mask": batch['attention_mask'].to(device),
            "start_positions": batch['answer_start_token'].to(device),
            "end_positions": batch['answer_end_token'].to(device),
        }

        outputs = best_model(**inputs)
        loss = outputs[0].mean()
        epoch_test_loss += loss.item()

    # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
    for i, start_output_logits in enumerate(outputs['start_logits']):
        start_output_logits = torch.where(inputs['token_type_ids'][i]!=0, outputs['start_logits'], inputs['token_type_ids'][i]-10000)
        end_output_logits = torch.where(inputs['token_type_ids'][i]!=0, outputs['end_logits'], inputs['token_type_ids'][i]-10000)

    start_preds = torch.argmax(start_output_logits, dim=1)
    end_preds = torch.argmax(end_output_logits, dim=1)

    # inspect predictions and answers
    for i in range(len(start_preds)):
        print(tokenizer.decode(inputs['input_ids'][i][start_preds[i]:end_preds[i]]))
        print(tokenizer.decode(inputs['input_ids'][i][inputs['start_positions'][i]:inputs['end_positions'][i]]))
        print('-----------')

    pred_batch = [el for el in zip(start_preds.tolist(), end_preds.tolist())]
    true_batch = [el for el in zip(inputs["start_positions"].tolist(), inputs["end_positions"].tolist())]

    test_all_preds.extend(pred_batch)
    test_all_true.extend(true_batch)

    pred_batch_ids = [str(uuid.uuid4()) for i in range(len(start_preds))]

    for i, pair in enumerate(pred_batch):
        if pair[0] >= pair[1]:
            text_pred = ''
        else:
            text_pred = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
            if not isinstance(text_pred, str):
                text_pred = ''
        entry = {
            'prediction_text': text_pred,
            'id': pred_batch_ids[i],
            'no_answer_probability': 0
        }

        test_qa_preds.append(entry)
            
        for i, pair in enumerate(true_batch):
            text_true = tokenizer.decode(inputs['input_ids'][i][pair[0]:pair[1]])
            entry = {
                'answers': {
                    'answer_start': [true_batch[0][0]],
                    'text': [text_true],
                    },
                    'id': pred_batch_ids[i]
                }
                
        test_qa_trues.append(entry)

results_test_squad_evaluate = squad_metric_evaluate.compute(predictions=test_qa_preds, references=test_qa_trues)
results_test_squad_datasets = squad_metric_datasets.compute(predictions=test_qa_preds, references=test_qa_trues)

epoch_test_loss /= len(test_loader)
test_losses.append(epoch_test_loss)

# Calculate testidation metrics
test_pred_flat = [p for pair in test_all_preds for p in pair]
test_true_flat = [t for pair in test_all_true for t in pair]

test_f1 = f1_score(test_true_flat, test_pred_flat, average=avg_type)
test_exact_match = accuracy_score(test_true_flat, test_pred_flat)
test_f1s.append(test_f1)
test_exact_matches.append(test_exact_match)
test_f1s_squad_evaluate.append(results_test_squad_evaluate['f1'])
test_exact_matches_squad_evaluate.append(results_test_squad_evaluate['exact'])
test_f1s_squad_datasets.append(results_test_squad_datasets['f1'])
test_exact_matches_squad_datasets.append(results_test_squad_datasets['exact_match'])

# Log Epoch Metrics
print(
        # "\nTest Loss:", test_losses[-1],
        # f"\nTest F1 {avg_type}:", test_f1s[-1],
        # "\Test Exact Match:", test_exact_matches[-1],
        "\nTest F1 Squad Evaluate:", results_test_squad_evaluate['f1'],
        "\nTest Exact Squad Evaluate:", results_test_squad_evaluate['exact'],
        # "\nTest F1 Squad Datasets:", results_test_squad_datasets['f1'],
        # "\nTest Exact Squad Datasets:", results_test_squad_datasets['exact_match'],
        "\n-----------------------",
        "\n\n")

train_losses.append('')
train_f1s.append('')
train_exact_matches.append('')
train_f1s_squad_evaluate.append('')
train_exact_matches_squad_evaluate.append('')
train_f1s_squad_datasets.append('')
train_exact_matches_squad_datasets.append('')
val_losses.append('')
val_f1s.append('')
val_exact_matches.append('')
val_f1s_squad_evaluate.append('')
val_exact_matches_squad_evaluate.append('')
val_f1s_squad_datasets.append('')
val_exact_matches_squad_datasets.append('')

# Save metrics to DataFrame and CSV
dict_df = {
    # 'epoch': [el for el in range(int(model_name.split('-')[-2])+1)] + [epochs_best], 
    'epoch': model_name.split('/')[-1],
    'train_loss': train_losses[-1],
    # f'train_f1_{avg_type}': train_f1s[-1],
    # 'train_exact_match': train_exact_matches[-1],
    'train_f1s_squad_evaluate': train_f1s_squad_evaluate[-1],
    'train_exact_matches_squad_evaluate': train_exact_matches_squad_evaluate[-1],
    # 'train_f1s_squad_datasets': train_f1s_squad_datasets[-1],
    # 'train_exact_matches_squad_datasets': train_exact_matches_squad_datasets[-1],
    # 'val_loss': val_losses[-1],
    # f'val_f1_{avg_type}': val_f1s[-1],
    # 'val_exact_match': val_exact_matches[-1],
    'val_f1s_squad_evaluate': val_f1s_squad_evaluate[-1],
    'val_exact_matches_squad_evaluate': val_exact_matches_squad_evaluate[-1],
    # 'val_f1s_squad_datasets': val_f1s_squad_datasets[-1],
    # 'val_exact_matches_squad_datasets': val_exact_matches_squad_datasets[-1],
    # 'test_loss': test_losses[-1],
    # f'test_f1_{avg_type}': test_f1s[-1],
    # 'test_exact_match': test_exact_matches[-1],
    'test_f1s_squad_evaluate': test_f1s_squad_evaluate[-1],
    'test_exact_matches_squad_evaluate': test_exact_matches_squad_evaluate[-1],
    # 'test_f1s_squad_datasets': test_f1s_squad_datasets[-1],
    # 'test_exact_matches_squad_datasets': test_exact_matches_squad_datasets[-1],
}

df = pd.concat([df, pd.DataFrame(dict_df, index=[0])])

df.to_csv(os.path.join(results_path, csv_filename), index=False)




