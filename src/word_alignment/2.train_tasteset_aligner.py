import json
from word_alignment.utils import WADataset

with open('/home/pgajo/working/food/data/TASTEset/data/entity-wise/EW-TASTE_en-it_DEEPL.json') as f:
    data = json.load(f)
recipe_list = data['annotations']
print(len(recipe_list))

dataset_list = []
max_len = 0

# the end is shifted by 1 compared to the start because the model predicts the token to the left of the end of the answer (e.g., 7:7 instead of 7:8)
shift_type = {
    'bert': (-1, 0),
    'deberta': (-1, 0),
    'roberta': (0, 1),
}

shifts = shift_type['bert']

import re
def remove_non_alphanumeric(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s)

src_context_flag = 1

for i, recipe in enumerate(recipe_list):
    for j, entity in enumerate(recipe['entities_en']):
        entry = {}
        entry['id_recipe'] = i
        entry['id_entity'] = j

        if src_context_flag:
            entry['query'] = recipe['text_en'][:entity[0]] + '• ' + recipe['text_en'][entity[0]:entity[1]] + ' •' + recipe['text_en'][entity[1]:]
        else:
            entry['query'] = recipe['text_en'][entity[0]:entity[1]]
        
        entry['context'] = recipe['text_it']
        entry['answer'] = recipe['text_it'][recipe['entities_it'][j][0]:recipe['entities_it'][j][1]]
        entry['answer_start'] = recipe['entities_it'][j][0]
        entry['answer_end'] = recipe['entities_it'][j][1]
        
        query_encoding = tokenizer(entry['query'])
        context_encoding = tokenizer(entry['context'])

        entry['answer_start_token'] = context_encoding.char_to_token(entry['answer_start']) + len(query_encoding['input_ids']) + shifts[0]
        entry['answer_end_token'] = context_encoding.char_to_token(entry['answer_end'] - 1) + len(query_encoding['input_ids']) + shifts[1]

        input_encoding = tokenizer(entry['query'],
                                   entry['context'],#.replace(";", "|").replace(" ", "|"),
                                   )

        if max_len < len(input_encoding['input_ids']):
            max_len = len(input_encoding['input_ids'])
        
        char_check = entry['context'][entry['answer_start']:entry['answer_end']]
        token_check_encoded = input_encoding['input_ids'][entry['answer_start_token']:entry['answer_end_token']]
        token_check = tokenizer.decode(input_encoding['input_ids'][entry['answer_start_token']:entry['answer_end_token']])
        # print('query', entry['query'])
        # print('context', entry['context'])
        # print('answer', entry['answer'])
        # print(token_check)

        if remove_non_alphanumeric(char_check) != remove_non_alphanumeric(token_check):
            print(i)
            print('ERROR: char_check != token_check')
            print("entry['id_recipe']", entry['id_recipe'])
            print("entry['id_entity']", entry['id_entity'])
            print("entry['query']", [entry['query']])
            print("entry['context']", entry['context'])
            print("entry['answer_start']", entry['answer_start'])
            print("entry['answer_end']", entry['answer_end'])
            print("char_check", [char_check])

            print('full encoding:', input_encoding['input_ids'])
            print('query encoding:', query_encoding['input_ids'])
            print('context encoding:', context_encoding['input_ids'])
            print([f"{i}, {tokenizer.decode([el])}" for i, el in enumerate(input_encoding['input_ids'])])
            print(input_encoding['input_ids'][entry['answer_start_token']:entry['answer_end_token']])

            print("entry['answer_start_token']", entry['answer_start_token'])
            print("entry['answer_end_token']", entry['answer_end_token'])
            print("token_check_encoded", token_check_encoded)
            print("token_check", [token_check])

            print('-------------------------')
            continue
        dataset_list.append(entry)
print('max_len', max_len)

import pandas as pd
dataset_df = pd.DataFrame(dataset_list).drop_duplicates(['answer'])
dataset_list = dataset_df.to_dict('records')
len(dataset_list)
dataset_df.head()

from datasets import Dataset, DatasetDict
dataset_unsplit = Dataset.from_list(dataset_list)

train_test_split = dataset_unsplit.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})


# print(dataset)
def tokenize_function(example):
    return tokenizer(
        example['query'],
        example['context'],
        padding='max_length',
        truncation=True,
        max_length=max_len
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format('torch', columns=['input_ids',
                                               'token_type_ids',
                                                'attention_mask',
                                                'answer_start_token',
                                                'answer_end_token'])
print(tokenized_dataset)


import torch
train_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size = 8, shuffle = True)
val_loader = torch.utils.data.DataLoader(tokenized_dataset['validation'], batch_size = 8, shuffle = True)

from transformers import AutoModelForQuestionAnswering
import torch
torch.set_printoptions(linewidth=1000)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
model = torch.nn.DataParallel(model)  # Use DataParallel
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training setup
from tqdm.auto import tqdm
import time
current_timeanddate = time.strftime("%Y%m%d-%H%M%S")
previous_epochs = 0
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os

results_path = '/home/pgajo/working/food/src/word_alignment/TASTEset_recipe_alignment/results'

# Lists to store metrics
train_losses, train_f1s, train_exact_matches, train_f1s_squad_evaluate, train_exact_matches_squad_evaluate, train_f1s_squad_datasets, train_exact_matches_squad_datasets = [], [], [], [], [], [], []
val_losses, val_f1s, val_exact_matches, val_f1s_squad_evaluate, val_exact_matches_squad_evaluate, val_f1s_squad_datasets, val_exact_matches_squad_datasets = [], [], [], [], [], [], []
test_losses, test_f1s, test_exact_matches, test_f1s_squad_evaluate, test_exact_matches_squad_evaluate, test_f1s_squad_datasets, test_exact_matches_squad_datasets = [], [], [], [], [], [], []

avg_type = 'micro'

print_every = 100

import uuid
from evaluate import load
squad_metric_evaluate = load("squad_v2")
from datasets import load_metric
squad_metric_datasets = load_metric("squad")

src_context_opt = 'with_src_context' if src_context_flag else 'no_src_context'

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

    # Training
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

    # Validation
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

    # Calculate evaluation metrics
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
    csv_filename = f"{current_timeanddate}_{model_name.split('/')[-1]}_metrics_{src_context_opt}.csv"
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

model.eval()
# print validation outputs for the first batches just to make sure this is working
for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
    # if batch_idx > 10:
    #     break
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

    for i in range(len(pred_batch)):
        text_pred = tokenizer.decode(inputs['input_ids'][i][pred_batch[i][0]:pred_batch[i][1]])
        text_true = tokenizer.decode(inputs['input_ids'][i][true_batch[i][0]:true_batch[i][1]])
        if text_pred != text_true:
            print(text_pred)
            print(text_true)
            print('--------')

# try aligning sentence-wise translated TASTEset

import json
from tqdm.auto import tqdm
bilingual_path = '/home/pgajo/working/food/data/TASTEset/data/formatted data/TASTEset_sep_format_en-it_unaligned.json'
with open(bilingual_path, encoding='utf8') as f:
    data = json.load(f)

print(data['annotations'][:3])

recipe_list = data['annotations']#[:3] 
len(recipe_list)

for idx, recipe in tqdm(enumerate(recipe_list), total=len(recipe_list)):

    for i, entity in enumerate(recipe['entities_en']):

        with torch.inference_mode():
            
            input = tokenizer(
                recipe['text_en'][:entity[0]] + '• ' + recipe['text_en'][entity[0]:entity[1]] + ' •' + recipe['text_en'][entity[1]:],
                recipe['text_it'],
                return_tensors='pt',
            ).to('cuda')

            input_ids = input['input_ids'].squeeze()
            
            outputs = model(**input)

            # set to -10000 any logits in the query (left side of the input) so that the model cannot predict those tokens
            for i in range(len(outputs['start_logits'])):
                outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], input['token_type_ids'][i]-10000)
                outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], input['token_type_ids'][i]-10000)

            start_preds = torch.argmax(outputs['start_logits'], dim=1)
            end_preds = torch.argmax(outputs['end_logits'], dim=1)
        # print("recipe['text_it']", recipe['text_it'])
        decoded_input = tokenizer.decode(input_ids)
        # print('decoded:', decoded_input)

        start_index_token = start_preds
        # print('start_index_token', start_index_token)
        end_index_token = end_preds
        # print('end_index_token', end_index_token)
        # print('len(input_ids)', len(input_ids))
        if start_index_token >= len(input_ids) - 1 or end_index_token >= len(input_ids) - 1:
            continue
        # print('encoding:', input_ids)
        
        # for j, id in enumerate(input_ids):
        #     print(j, int(id), tokenizer.decode([id]), end='\t\t')
        # print()
        # print('prediction_tokens:', input['input_ids'].squeeze()[start_index_token:end_index_token])
        # print('prediction:', tokenizer.decode(input['input_ids'].squeeze()[start_index_token:end_index_token]))
        # print('gold:', [recipe['text_en'][entity[0]:entity[1]]])
        char_span_start = input.token_to_chars(start_index_token)
        # print('char_span_start', char_span_start)
        char_span_end = input.token_to_chars(end_index_token-1)
        char_span_prediction = recipe['text_it'][char_span_start[0]:char_span_end[1]]
        char_span_prediction_splitjoined = ''.join(char_span_prediction.split()).replace('#', '')
        # print('char_span_prediction', )
        # print('char_span_end', char_span_end)
        char_span = (char_span_start[0], char_span_end[1])
        # print('char_span', char_span)
        # print('char_span[0]', char_span[0])
        # print('char_span[1]', char_span[1])
        if not char_span[0] > char_span[1]:
            recipe['entities_it'].append([char_span[0], char_span[1], recipe['entities_en'][i][2]])
            # print(recipe['entities_it'])
        # else:
            # print('skipping')
        
        token_based_prediction = tokenizer.decode(input['input_ids'].squeeze()[start_index_token:end_index_token])
        token_based_prediction_splitjoined = ''.join(token_based_prediction.split()).replace('#', '')
        gold = recipe['text_en'][entity[0]:entity[1]]
        gold_splitjoined = ''.join(gold.split()).replace('#', '')
        if gold_splitjoined != token_based_prediction_splitjoined:
            print('source:', recipe['text_en'][:entity[0]] + '• ' + recipe['text_en'][entity[0]:entity[1]] + ' •' + recipe['text_en'][entity[1]:])
            print('target:', recipe['text_it'])
            # print('encoding:', input_ids)
            # print('decoded:', decoded_input)
            # print('prediction_tokens:', input['input_ids'].squeeze()[start_index_token:end_index_token])
            # print('char_span_prediction:', [char_span_prediction])
            # print('char_span_prediction_splitjoined:', [char_span_prediction_splitjoined]) 
            print('token_based_prediction:', [token_based_prediction])
            # print('token_based_prediction_splitjoined:', [token_based_prediction_splitjoined])
            print('gold:', [gold])
            print('---------------------------')

        if char_span_prediction_splitjoined != token_based_prediction_splitjoined:
            print('ERROR: CHAR SPAN PREDICTION DOES NOT MATCH TOKEN-BASED PREDICTION')
            # print(f'################# SKIPPING #################\nerrors: {error_count+1}\nerror rate = {error_count/sample_count}')
            error = True
            print(f'recipe no.: {idx}, entity no.: {i}')
            print(recipe['text_en'][entity[0]:entity[1]])
            print(recipe['text_it'])
            print('start_index_token:', start_index_token)
            print('end_index_token:', end_index_token)
            print('len(input_ids):', len(input_ids))
            print('encoding:', input_ids)
            print('decoded:', decoded_input)
            print('prediction_tokens:', input['input_ids'].squeeze()[start_index_token:end_index_token])
            print('token_based_prediction:', [token_based_prediction])
            print('token_based_prediction_splitjoined:', [token_based_prediction_splitjoined])
            print('gold:', [recipe['text_en'][entity[0]:entity[1]]])
            print('char_span_start', char_span_start)
            print('char_span_end', char_span_end) 
            print('char_span_prediction:', [char_span_prediction])
            print('char_span_prediction_splitjoined:', [char_span_prediction_splitjoined])
            print('full context_en:', recipe['text_en'])
            print('full context_it:', recipe['text_it'])
            break
print(data)

from huggingface_hub import login
token="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"
login(token=token)
model_save_name = f"pgajo/{model_name.split('/')[-1]}-recipe-aligner-en-it-{epochs_best}-epochs"
model.module.push_to_hub(model_save_name)

from huggingface_hub import whoami, create_repo, ModelCard, ModelCardData

user = whoami(token=token)
repo_id = model_save_name
url = create_repo(repo_id, exist_ok=True)
card_data = ModelCardData(language='en', license='mit', library_name='pytorch')
card = ModelCard.from_template(
    card_data,
    model_id=model_name.split('/')[-1],
    model_description="Recipe alignment model first fine-tuned on XL-WA (en-it combination only), and then fine-tuned on entity-wise machine-translated (DeepL) TASTEset recipes.",
    developers="Paolo Gajo",
    repo="https://github.com/paolo-gajo/food",
)
card.push_to_hub(repo_id)


