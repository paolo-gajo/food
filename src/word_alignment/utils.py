import os
import json
import pandas as pd
from typing import List, Union
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
import random
from torch.utils.data import DataLoader
import uuid
import torch
from huggingface_hub import login, ModelCard, ModelCardData
from datetime import datetime

class SquadEvaluator:
    '''
    Prepares evaluation samples for the Squad v2 Evaluate metric.
    
    Example:

    from evaluate import load
    squad_metric_evaluate = load("squad_v2")
    evaluator = SquadEvaluator()
    outputs = model(**batch)
    evaluator.get_eval_batch(outputs)
    '''

    def __init__(self,
                tokenizer,
                model,
                eval_fc,
                patience = 2
                ) -> None:
        self.preds = {'train': [], 'dev': [], 'test': []}
        self.trues = {'train': [], 'dev': [], 'test': []}

        self.metrics = []
        self.epoch_metrics = {
            'train_loss': 0, 'train_f1': 0, 'train_exact': 0,
            'dev_loss': 0, 'dev_f1': 0, 'dev_exact': 0,
            'test_loss': 0, 'test_f1': 0, 'test_exact': 0,
            }

        self.tokenizer = tokenizer
        self.eval_fc = eval_fc
        self.epoch_best = 1
        self.f1_dev_best = 0
        self.exact_dev_best = 0
        self.patience = patience
        self.patience_counter = 0
        self.stop_training = False
        self.best_model = None

    def evaluate(self, model, split, epoch, eval_metric = 'dev'):
        self.epoch_metrics[f'{split}_f1'] = self.eval_fc.compute(predictions=self.preds[split],
                                                references=self.trues[split])['f1']
        self.epoch_metrics[f'{split}_exact'] = self.eval_fc.compute(predictions=self.preds[split],
                                                references=self.trues[split])['exact']
        if split == eval_metric:
            if self.epoch_metrics[f'{eval_metric}_exact'] > self.exact_dev_best:
                self.exact_dev_best = self.epoch_metrics[f'{eval_metric}_exact']
                self.epoch_best = epoch + 1
                self.patience_counter = 0
                print(f'----Best dev exact updated: {round(self.exact_dev_best, ndigits=2)}\
                    \n----Best epoch updated: {self.epoch_best}')
                if hasattr(model, 'module'):
                    self.best_model = model.module
                else:
                    self.best_model = model
            else:
                self.patience_counter += 1
                print(f'----Did not update best model, patience: {self.patience_counter}')
            
            if self.patience_counter > self.patience:
                self.stop_training = True

        return self.stop_training
    
    def store_metrics(self):
        self.metrics.append(self.epoch_metrics)
        self.epoch_metrics = {
            'train_loss': 0, 'train_f1': 0, 'train_exact': 0,
            'dev_loss': 0, 'dev_f1': 0, 'dev_exact': 0,
            'test_loss': 0, 'test_f1': 0, 'test_exact': 0,
            }

    def get_eval_batch(self, model_outputs, batch, split):
        start_preds = torch.argmax(model_outputs['start_logits'], dim=1)
        end_preds = torch.argmax(model_outputs['end_logits'], dim=1)
        pred_batch = [el for el in zip(start_preds.tolist(),
                                       end_preds.tolist())]
        true_batch = [el for el in zip(batch["start_positions"].tolist(),
                                       batch["end_positions"].tolist())]
        pred_batch_ids = [str(uuid.uuid4()) for i in range(len(start_preds))]

        for i, pair in enumerate(pred_batch):
            if pair[0] >= pair[1]:
                text_pred = ''
            else:
                text_pred = self.tokenizer.decode(batch['input_ids'][i][pair[0]:pair[1]])
                if not isinstance(text_pred, str):
                    text_pred = ''
            dict_pred = {
                'prediction_text': text_pred,
                'id': pred_batch_ids[i],
                'no_answer_probability': 0
            }

        for i, pair in enumerate(true_batch):
            text_true = self.tokenizer.decode(
                batch['input_ids'][i][pair[0]:pair[1]])
            dict_true = {
                'answers': {
                    'answer_start': [true_batch[0][0]],
                    'text': [text_true],
                    },
                    'id': pred_batch_ids[i]
                }

        self.preds[split].append(dict_pred)
        self.trues[split].append(dict_true)
    
    def print_metrics(self, current_epoch = None, current_split = None):
        if current_epoch is None:
            df = pd.DataFrame(self.metrics).round(decimals = 2)
            print(df)
            return df
        if current_split is not None:
            dict_current_split = {metric: self.epoch_metrics[metric] for metric in [f'{current_split}_loss', f'{current_split}_f1', f'{current_split}_exact']}
            df = pd.DataFrame.from_dict(dict_current_split, orient='index')
        else:
            df = pd.DataFrame.from_dict(self.epoch_metrics, orient='index')
        df = df.round(decimals = 2)
        df.index.name = f'epoch: {current_epoch}'
        print(df)
        return df
    
    def save_metrics_to_csv(self, path, add_date = True, add_best_dev = True, format = 'csv'):
        now = datetime.now()
        if add_date:
            dt_string = now.strftime("%Y-%m-%d_%H-%M-%S") + '_'
        else:
            dt_string = ''
        
        if add_best_dev:
            bl_string = '_' + str(self.exact_dev_best)
        else:
            bl_string = ''
        df = pd.DataFrame(self.metrics)
        df.index += 1
        df.index.name = 'epoch'
        if not os.path.isdir(path):
            os.makedirs(path)
        metrics_save_name = os.path.join(path, dt_string + self.best_model.config._name_or_path + bl_string)
        print(f'Saving metrics to: {metrics_save_name}')
        df.to_csv(metrics_save_name + f'.{format}')

def prep_xl_wa(data_path,
                languages,
                tokenizer,
                n_rows = None,
                splits = ['train', 'dev', 'test']
                ):
    '''
    Prepares data from the XL-WA dataset based on the wanted languages,
    tokenizer, number of rows and whether
    we want source context or not surrounding the query word.
    '''
    ds = {key: [] for key in splits}
    ds_dict_hf = DatasetDict()
    for split in splits:
        ds[split] = []
        unpacked_lines = []
        for lang in languages:
            df = pd.read_csv(os.path.join(data_path,
                                            f'{lang}/{split}.tsv'),
                                            sep='\t',
                                            header=None)
            df.columns = ['src', 'tgt', 'spans']
            df['lang'] = lang
            df['pairs'] = df.apply(
                lambda row: convert_alignments(row[0], row[1], row[2]), axis=1)
            unpacked_lines += df[:n_rows].to_dict(orient='records')
        progbar = tqdm(unpacked_lines, total = len(unpacked_lines))
        progbar.set_description(f'Creating samples for {split} split...')
        for line in progbar:
            ds[split] += create_samples(line, tokenizer)

        df_hf = Dataset.from_list(ds[split])
        ds_dict_hf[split] = df_hf
    return ds_dict_hf

def prep_tasteset(data_path,
                languages,
                tokenizer,
                shuffle_num = None,
                shuffle = True,
                n_rows = None,
                splits = ['train', 'dev', 'test']
                ):
    '''
    Prepares data from the TASTEset dataset based on the wanted languages,
    tokenizer, number of rows and whether
    we want source context or not surrounding the query word.
    '''
    with open(data_path) as f:
        data = json.load(f)
    recipe_list = data['annotations']

    ds = {key: [] for key in splits}
    ds_dict_hf = DatasetDict()
    progbar = tqdm(recipe_list, total = len(recipe_list))
    progbar.set_description(f'Creating samples for {split} split...')
    
    for line in progbar:
        ds[split] += create_samples(line, tokenizer)

    df_hf = Dataset.from_list(ds[split])
    ds_dict_hf[split] = df_hf
    
    return ds_dict_hf

def qa_tokenize(sample: Union[str, List], tokenizer):
    '''
    Pass this to .map when tokenizing a dataset for QA-style training.
    Remember to shuffle before using this, otherwise if you shuffle
    after you get batch size mismatches,
    since we're using longest in the batch for padding.
    '''
    sample.update(tokenizer(sample['query'],
                    sample['context'],
                    padding = 'longest',
                    truncation = True,
                    return_tensors = 'pt'
                    ))
    return sample

def create_samples(sample, tokenizer, src_context = True):
    '''
    Creates one sample per word in the source sentence of the sample.
    '''
    sample_list = []
    for j, pair in enumerate(sample['pairs']):
        new_sample = {}

        if src_context:
            new_sample['query'] = sample['src'][:pair['src_start']] +\
                        '• ' + sample['src'][pair['src_start']:pair['src_end']] + ' •' +\
                        sample['src'][pair['src_end']:]
        else:
            new_sample['query'] = sample['src'][pair['src_start']:pair['src_end']]

        new_sample['context'] = sample['tgt']
        new_sample['answer'] = sample['tgt'][pair['tgt_start']:pair['tgt_end']]
        new_sample['answer_start'] = pair['tgt_start']
        new_sample['answer_end'] = pair['tgt_end']
        query_enc = tokenizer(new_sample['query'])
        l = len(query_enc['input_ids']) 
        context_enc = tokenizer(new_sample['context'])

        # start_positions and end_positions are token positions
        new_sample['start_positions'] = context_enc.char_to_token(
            new_sample['answer_start']) - 1 + l
        new_sample['end_positions'] = context_enc.char_to_token(
            new_sample['answer_end']-1) + l
        sample_list.append(new_sample)
    return sample_list

def convert_alignments(src_sentence, tgt_sentence, alignments):
    '''
    Converts GIZA-style '0-1' string alignments to dicts like:
        {
            'src_start': src_span[0],
            'src_end': src_span[1],
            'tgt_start': tgt_span[0],
            'tgt_end': tgt_span[1],
        }
    '''
    src_spans = calculate_spans(src_sentence)
    tgt_spans = calculate_spans(tgt_sentence)
    converted_alignments = []
    for alignment in alignments.split():
        src_idx, tgt_idx = map(int, alignment.split('-'))
        src_span = src_spans[src_idx]
        tgt_span = tgt_spans[tgt_idx]

        converted_alignments.append({
            'src_start': src_span[0],
            'src_end': src_span[1],
            'tgt_start': tgt_span[0],
            'tgt_end': tgt_span[1],
        })
    return converted_alignments

def calculate_spans(sentence):
    spans = []
    start = 0

    for word in sentence.split():
        end = start + len(word)
        spans.append((start, end))
        start = end + 1
    return spans
    
def shuffle_entities(sample, lang):
    '''
    Shuffles entities and text for a sample.
    Only shuffles, so the source and target entities need 
    to be linked somehow after shuffling one language
    (usually the target language)!
    '''
    text_key = f'text_{lang}'
    ent_key = f'entities_{lang}'
    shuffled_ents = []
    indexes = [i for i in range(len(sample[ent_key]))]
    random.shuffle(indexes)

    # get non-entity positions and strings from the original text
    blanks = []
    for i in range(len(sample[ent_key])-1):
        end_prev = sample[ent_key][i][1]
        start_foll = sample[ent_key][i+1][0]
        blanks.append([end_prev, start_foll, sample[text_key][end_prev:start_foll]])
        
    ent_start = 0
    shuffled_text = ''
    for new, idx in enumerate(indexes):
        tmp_ent = sample[ent_key][idx]
        text_tmp_ent = sample[text_key][sample[ent_key][idx][0]:sample[ent_key][idx][1]]
        
        len_text = len((text_tmp_ent))
        tmp_ent[0] = ent_start
        tmp_ent[1] = tmp_ent[0] + len_text
        tmp_ent[2] = sample[ent_key][idx][2]
        shuffled_ents.append(tmp_ent)

        if len(blanks) > 0:
            next_blank = blanks.pop(0)
            ent_start += len((text_tmp_ent)) + next_blank[1] - next_blank[0]
        else:
            pass

        shuffled_text += text_tmp_ent + next_blank[2]
    shuffled_sample = {
        text_key: shuffled_text,
        ent_key: shuffled_ents
    }
    return shuffled_sample

def data_loader(dataset, batch_size, n_rows = None):
    loader_train = DataLoader(dataset['train'].select(
                            range(len(dataset['train']))[:n_rows]),
                            batch_size = batch_size,
                            # shuffle = True
                            )
    loader_dev = DataLoader(dataset['dev'].select(
                            range(len(dataset['dev']))[:n_rows]),
                            batch_size = batch_size,
                            # shuffle = True
                            )
    loader_test = DataLoader(dataset['test'].select(
                            range(len(dataset['test']))[:n_rows]),
                            batch_size = batch_size,
                            # shuffle = True
                            )
    return {'train': loader_train,
            'dev': loader_dev,
            'test': loader_test}

def push_model(model,
               model_name = None,
               user = 'pgajo/',
               suffix = '',
               model_description = '',
               language = 'en',
               repo = "https://github.com/paolo-gajo/food",
               ):
    # save best model
    if hasattr(model, 'module'):
        model = model.module
        
    token='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'
    login(token=token)
    if model_name is None:
        model_name = model.config._name_or_path
    model_save_name = user + model_name + suffix
    model.push_to_hub(model_save_name)
    
    # user = whoami(token=token)
    repo_id = model_save_name
    # url = create_repo(repo_id, exist_ok=True)
    card_data = ModelCardData(language=language,
                              license='mit',
                              library_name='pytorch')
    card = ModelCard.from_template(
        card_data,
        model_id = model_name.split('/')[-1],
        model_description = model_description,
        developers = "Paolo Gajo",
        repo = repo,
    )
    card.push_to_hub(repo_id)

