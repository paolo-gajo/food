import os
import json
import pandas as pd
from typing import List, Union
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
import random
from torch.utils.data import DataLoader
import uuid
import torch
from huggingface_hub import ModelCard, ModelCardData, DatasetCard, DatasetCardData
from datetime import datetime
import copy
from transformers import AutoTokenizer

sep_dict = {
    'csv': ',',
    'tsv': '\t'
}

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
        self.epoch_best = 0
        self.f1_dev_best = 0
        self.exact_dev_best = 0
        self.patience = patience
        self.patience_counter = 0
        self.stop_training = False
        self.best_model = model

    def evaluate(self, model, split, epoch, eval_metric = 'dev'):
        print(f'len(self.preds[{split}])', len(self.preds[split]))
        self.epoch_metrics[f'{split}_f1'] = self.eval_fc.compute(predictions=self.preds[split],
                                                references=self.trues[split])['f1']
        self.epoch_metrics[f'{split}_exact'] = self.eval_fc.compute(predictions=self.preds[split],
                                                references=self.trues[split])['exact']

        self.preds[split] = []
        self.trues[split] = []

        if split == eval_metric:
            if self.epoch_metrics[f'{eval_metric}_exact'] > self.exact_dev_best:
                self.exact_dev_best = self.epoch_metrics[f'{eval_metric}_exact']
                self.epoch_best = epoch + 1
                self.patience_counter = 0
                print(f'----Best {eval_metric} exact updated: {round(self.exact_dev_best, ndigits=2)}\
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
        self.split = split
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

            self.preds[split].append(dict_pred)

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
        df.index.name = f'epoch: {current_epoch + 1}'
        print(df)
        return df
    
    def save_metrics_to_csv(self, path, add_date = True, add_best_dev = True, format = 'tsv'):
        now = datetime.now()
        if add_date:
            dt_string = now.strftime("%Y-%m-%d_%H-%M-%S") + '_'
        else:
            dt_string = ''
        
        if add_best_dev:
            exact_dev_best_suffix = '_' + str(round(self.exact_dev_best, ndigits=2))
        else:
            exact_dev_best_suffix = ''
        df = pd.DataFrame(self.metrics)
        df.index += 1
        df.index.name = 'epoch'
        if not os.path.isdir(path):
            os.makedirs(path)
        if self.epoch_best > 0 or self.split == 'test':
            metrics_save_name = os.path.join(path, dt_string + self.best_model.config._name_or_path.split('/')[-1] + exact_dev_best_suffix)
            print(f'Saving metrics to: {metrics_save_name}')
            csv_save_name = metrics_save_name + f'.{format}'
            df.to_csv(csv_save_name, sep=sep_dict[format])
        else:
            print('No best model! Did you run with too few instances just for testing?')
    
    def append_test_metrics(self, path, format = 'tsv'):
        csv_path = os.path.join(path, 'metrics') + f'.{format}'
        df_orig = pd.read_csv(csv_path)
        df_append = pd.DataFrame(self.metrics, index = [0])
        df_append['epoch'] = 'test'
        df = pd.concat([df_orig, df_append])
        df.to_csv(csv_path, sep=sep_dict[format], index=False)

class SampleList:
    def __init__(self, samples:List[dict], shuffle:bool = False) -> None:
        self.samples = samples
        self.shuffle = shuffle
        self.index = None

class TASTEset(DatasetDict):
    def __init__(self,
        input_data,
        *,
        tokenizer_name = '',
        data_path = '',
        src_lang = 'en',
        tgt_langs = ['it'],
        drop_duplicates = True,
        src_context = True,
        shuffled_size = 0,
        unshuffled_size = 1,
        dev_size = 0.2,
        batch_size = None,
        debug_dump = False,
        aligned = True,
        n_rows = None,
        label_studio = False,
        inverse_languages = False,
        keep_raw = False,
        shuffle_type = 'recipe',
        ) -> 'TASTEset':
        '''
        Prepares data from the TASTEset dataset based on the wanted languages,
        tokenizer, number of rows and whether
        we want source context or not surrounding the query word.

        Args:

            input_data (`List[dict]`):
                List of dict samples.
            tokenizer_name (`str`):
                Hugging-face name of the tokenizer.
            src_lang (`str`):
                ISO 639-1 language code. Indicates which language is the source language,
                with the entities in that language not being shuffled.
                string of multiple space-separated language codes, or list of language codes.
            tgt_langs (`str` or `List[str]`):
                String of a single 2-character language code (ISO 639-1),
                string of multiple space-separated language codes, or list of language codes
                indicating the target languages of the dataset.
                The entities of the target languages will be shuffled in the portion of the dataset
                dictated by the size ratio 'shuffled_size'.
            drop_duplicates (`bool`, defaults to `True`):
                If `True`, drop duplicates from the dataset based on the ['answer'] column.
            src_context (`bool`, defaults to `True`):
                if `True`, the query is the whole source sentence and the word being aligned
                is surrounded with "•" characters.
            shuffled_size (`float`):
                Size of the dev split compared to the train split.
            shuffled_size (`int` or `float`):
                Controls the % size of the output shuffled sample 
                compared to the total shuffled samples being created.
            unshuffled_size (`int` or `float`):
                Controls the % size of the output unshuffled sample
                compared to the total unshuffled samples being created.
            dev_size (`int` or `float`):
                Size of the dev split compared to the train split.
            batch_size (`int`, defaults to `None`):
                Size of the train and dev batches.
                If None, the whole dataset is tokenized to the token length of the longest sample.
            aligned (`bool`, defaults to `True`):
                If `True`, the output dataset will include answer data,
                otherwise if `False` only 'query' and 'context' lines will be available for each sample.
                Set to `False` to create an unannotated dataset, for annotation.
                Set to `True` to make a training dataset when starting from an already annotated dataset.
            n_rows (`int`, defaults to `None`):
                Defines how many lines are going to be kept in the dataset. Normally used to speed up testing.
            label_studio (`bool`, defaults to `False`):
                If `True`, the input data is assumed to be in Label Studio format
                and is going to be converted to TASTEset format using 'label_studio_to_tasteset'.
            inverse_languages (`bool`, defaults to `False`):
                If `True`, check relations in the direction opposite to the the one they were annotated as,
                e.g. to allow producing Italian samples from English samples when relation annotations
                were actually done from Italian to English.
            keep_raw (`bool`, defaults to `False`):
                If `True`, keep raw text data in the dataset.
        '''
        self.name = 'TASTEset'
        columns = None
        if not keep_raw:
            columns = ['input_ids',
                        'token_type_ids',
                        'attention_mask',
                        'start_positions',
                        'end_positions']
        if isinstance(input_data, DatasetDict):
            input_data.set_format('torch', columns = columns)
            for key in input_data.keys():
                self[key] = input_data[key]

        else:
            self.src_lang = src_lang
            self.tgt_langs = tgt_langs
            self.inverse_languages = inverse_languages
            if label_studio:
                input_data = self.label_studio_to_tasteset(input_data)
            self.input_data = input_data['annotations']
            self.data_path = data_path
            self.tokenizer_name = tokenizer_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.rm_char = [' ', ';']
            
            self.drop_duplicates = drop_duplicates
            self.src_context = src_context
            self.shuffled_size = shuffled_size
            self.unshuffled_size = unshuffled_size
            self.dev_size = dev_size
            self.debug_dump = debug_dump
            self.aligned = aligned
            self.n_rows = n_rows
            self.unshuffled_samples = []
            self.shuffled_samples = []
            self.index = 0
            self.shuffled_type = shuffle_type
            if not hasattr(self.tokenizer, 'sep'):
                self.tokenizer.sep = None

            if isinstance(self.tgt_langs, str):
                self.tgt_langs = self.tgt_langs.split()

            if not self.aligned:
                self.drop_duplicates = False
                print("Input data is unaligned, setting 'drop_duplicates' to False.")
            
            self.prep_data()

            if self.aligned:
                dataset = self.map(
                    lambda sample: self.qa_tokenize(sample),
                    batched = True,
                    batch_size = batch_size,
                    )
                
                for key in dataset.keys():
                    self[key] = dataset[key]
                
                self.set_format('torch', columns = ['input_ids',
                                            'token_type_ids',
                                            'attention_mask',
                                            'start_positions',
                                            'end_positions']
                                            )
        
    @classmethod
    def from_json(cls,
                  json_path,
                  tokenizer_name,
                  **kwargs
                  ):
        input_data = json.load(open(json_path))
        return cls(
            input_data,
            data_path = json_path,
            tokenizer_name = tokenizer_name,
            **kwargs
        )
    
    @classmethod
    def from_datasetdict(cls,
                         repo_id,
                         **kwargs
                         ):
        data = load_dataset(repo_id)
        return cls(data,
                   **kwargs)

    def prep_data(self) -> 'DatasetDict':
        if self.unshuffled_size:
            self.unshuffled_samples = self.generate_samples(SampleList(self.extend(self.input_data[:self.n_rows], self.unshuffled_size), shuffle = False))
        if self.shuffled_size:
            self.shuffled_samples = self.generate_samples(SampleList(self.extend(self.input_data[:self.n_rows], self.shuffled_size), shuffle = True))
        df = pd.concat([pd.DataFrame(self.unshuffled_samples),
                        pd.DataFrame(self.shuffled_samples)])
        if self.drop_duplicates:
            df = df.drop_duplicates(['answer'])
        df_list = df.to_dict(orient = 'records')
        print('Number of samples:', len(df_list))
        if self.debug_dump:
            if self.aligned:
                dump_suffix = 'aligned_qa'
            else:
                dump_suffix = 'unaligned_qa'
            json.dump(df_list, open(self.data_path.replace('.json', f'_{dump_suffix}_dump.json'), 'w'), ensure_ascii=False)
        if self.dev_size == 0:
            ds_dict = Dataset.from_list(df_list)
            self['test'] = ds_dict
        else:
            ds_dict = Dataset.from_list(df_list).train_test_split(test_size = self.dev_size)
            self['train'] = ds_dict['train']
            ds_dict['dev'] = ds_dict.pop('test')
            self['dev'] = ds_dict['dev']

    def extend(self, sample_list, extend_ratio = 1):
        int_ratio = int(extend_ratio)
        decimals = extend_ratio % 1
        main = [el for el in sample_list * int_ratio]
        remainder = [el for el in sample_list[:round(len(sample_list) * decimals)]]
        extended_list = main + remainder

        # Creating a new list with new dictionary objects
        new_extended_list = []
        for i, el in enumerate(extended_list):
            new_el = {}
            new_el.update(el)  # create a new dictionary
            new_el['index'] = i
            new_extended_list.append(new_el)

        return new_extended_list

    def generate_samples(self, sample_list):
        shuffle_desc = 'shuffled' if sample_list.shuffle else 'unshuffled'
        progbar = tqdm(enumerate(sample_list.samples), total = len(sample_list.samples))
        progbar.set_description(f'Creating samples ({shuffle_desc})...')
        list_buffer = []
        for sentence_index, line in progbar:
            line_buffer = copy.deepcopy(line)
            list_buffer += self.samples_from_line(line_buffer, sentence_index, shuffle=sample_list.shuffle)
        return list_buffer

    def samples_from_line(self,
                        sample,
                        sentence_index,
                        *,
                        shuffle,
                        ) -> List[dict]:
        '''
        Creates one sample per word in the source sentence of the sample.
        '''
        sample = self.assign_indexes_to_entities(sample)
        tgt_lang = ''.join([lang for lang in sample['sample_langs'] if lang != self.src_lang])
        sample_list = []
        if shuffle:
            for shuffle_lang in self.tgt_langs:
                if self.shuffled_type == 'recipe':
                    sample = self.shuffle_entities_recipe(sample, shuffle_lang)
                if self.shuffled_type == 'ingredient':
                    sample = self.shuffle_entities_ingredient(sample, shuffle_lang)
        for idx in range(sample['num_ents']):
            new_sample = {}

            if self.src_context:
                new_sample['query'] = sample[f'text_{self.src_lang}'][:sample[f'ents_{self.src_lang}'][idx][0]] +\
                            '• ' + sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][idx][0]:sample[f'ents_{self.src_lang}'][idx][1]] + ' •' +\
                            sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][idx][1]:]
            else:
                new_sample['query'] = sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][0]:sample[f'ents_{self.src_lang}'][1]]

            new_sample['context'] = sample[f'text_{tgt_lang}']
            if self.aligned:
                new_sample['answer_start'] = sample[f'ents_{tgt_lang}'][sample[f'idx_{tgt_lang}'].index(idx)][0]
                new_sample['answer_end'] = sample[f'ents_{tgt_lang}'][sample[f'idx_{tgt_lang}'].index(idx)][1]
                new_sample['answer'] = sample[f'text_{tgt_lang}'][new_sample['answer_start']:new_sample['answer_end']]
                if new_sample['answer'] == '':
                    print('ANSWER IS EMPTY, IS THIS RIGHT?')
                    print(new_sample)
                    continue
                query_enc = self.tokenizer(new_sample['query'])
                l = len(query_enc['input_ids']) 
                context_enc = self.tokenizer(new_sample['context'])

                # start_positions and end_positions are token positions
                new_sample['start_positions'] = context_enc.char_to_token(
                    new_sample['answer_start']) - 1 + l
                new_sample['end_positions'] = context_enc.char_to_token(
                    new_sample['answer_end']-1) + l
                
            if self.tokenizer.sep:
                for char in self.rm_char:
                    new_sample['query'] = new_sample['query'].replace(char, self.tokenizer.sep)
                    new_sample['context'] = new_sample['context'].replace(char, self.tokenizer.sep)
            new_sample['sentence_index'] = sentence_index
            new_sample['sample_index'] = idx
            new_sample['index'] = self.index
            self.index += 1
            sample_list.append(new_sample)
        return sample_list

    def assign_indexes_to_entities(self, sample) -> dict:
        sample_langs = list(set([key.split('_')[-1] for key in sample.keys() if '_' in key]))
        original_indexes = [i for i in range(len(sample[f'ents_{self.src_lang}']))]
        for l in sample_langs:
            sample.update({f'idx_{l}': original_indexes,
                        'sample_langs': sample_langs,
                        'num_ents': len(original_indexes)})
        return sample

    def qa_tokenize(self, sample):
        '''
        Pass this to .map when tokenizing a dataset for QA-style training.
        Remember to shuffle before using this, otherwise if you shuffle
        after you get batch size mismatches,
        since we're using longest in the batch for padding.
        '''
        sample.update(self.tokenizer(sample['query'],
                        sample['context'],
                        padding = 'longest',
                        truncation = True,
                        return_tensors = 'pt'
                        ))
        return sample
    
    def label_studio_to_tasteset(self, data):
        if self.inverse_languages:
            source_id = 'to_id' # check relations in the opposite verse of annotation
            target_id = 'from_id'
        else:
            source_id = 'from_id' # check relations in the verse of annotation
            target_id = 'to_id'
        formatted_data = []
        for sample in data:
            sample_labels_src = [el for el in sample['annotations'][0]['result'] if el['type'] == 'labels' and el['from_name'] == f'label_{self.src_lang}']
            ents_src = []
            relations = [el for el in sample['annotations'][0]['result'] if el['type'] == 'relation']
            if len(relations)>0:
                sample_dict = {}
                for tgt_lang in self.tgt_langs:
                    sample_labels_tgt = [el for el in sample['annotations'][0]['result'] if el['type'] == 'labels' and el['from_name'] == f'label_{tgt_lang}']
                    ents_tgt = [] 
                    for label_src in sample_labels_src:
                        src_id = label_src['id']
                        src_ent = [label_src['value']['start'], label_src['value']['end'], label_src['value']['labels'][0], label_src['value']['text']]
                        for relation in relations:
                            if relation[source_id] == src_id:
                                tgt_id = relation[target_id]
                                for label_src in sample_labels_tgt:
                                    if label_src['id'] == tgt_id:
                                        tgt_ent = [label_src['value']['start'], label_src['value']['end'], label_src['value']['labels'][0], label_src['value']['text']]
                        ents_src.append(src_ent)
                        ents_tgt.append(tgt_ent)
                    sample_dict.update({
                        f'text_{tgt_lang}': sample['data'][f'ingredients_{tgt_lang}'],
                        f'ents_{tgt_lang}': ents_tgt,
                    })
                sample_dict.update({
                    f'text_{self.src_lang}': sample['data'][f'ingredients_{self.src_lang}'],
                    f'ents_{self.src_lang}': ents_src,
                })
                formatted_data.append(sample_dict)
        return {'annotations': formatted_data}

    @staticmethod
    def shuffle_entities_ingredient(sample, shuffle_lang, verbose = False) -> dict:
        '''
        Shuffles entities and text for a sample.
        Only shuffles, so the source and target entities need 
        to be linked somehow after shuffling one language
        (usually the target language)!
        '''
        text_key = f'text_{shuffle_lang}'
        ent_key = f'ents_{shuffle_lang}'
        shuffled_ents = []
        shuffled_indexes = [i for i in range(len(sample[ent_key]))]
        random.shuffle(shuffled_indexes)
        sample.update({f'idx_{shuffle_lang}': shuffled_indexes})
        
        # get non-entity positions and strings from the original text
        blanks = []
        for i in range(len(sample[ent_key])-1):
            end_prev = sample[ent_key][i][1]
            start_foll = sample[ent_key][i+1][0]
            blanks.append([end_prev, start_foll, sample[text_key][end_prev:start_foll]])
            
        ent_start = 0
        shuffled_text = ''
        for new, idx in enumerate(sample[f'idx_{shuffle_lang}']):
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
        sample.update({
            text_key: shuffled_text,
            ent_key: shuffled_ents,
        })
        if verbose:
            print_list = []
            for i in range(len(sample[f'idx_{shuffle_lang}'])):
                row = []
                for l in sample['sample_langs']:
                    row.append([[sample[f'idx_{l}'].index(i)], sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)] + [sample[f'text_{l}'][sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)][0]:sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)][1]]]])
                print_list.append(row)
            print(pd.DataFrame(print_list))
            print(sample['idx_en'])
            print(sample['idx_it'])

        return sample

    @staticmethod
    def shuffle_entities_recipe(sample, shuffle_lang, verbose = False) -> dict:
        '''
        Shuffles entities and text for a sample.
        Only shuffles, so the source and target entities need 
        to be linked somehow after shuffling one language
        (usually the target language)!
        '''
        text_key = f'text_{shuffle_lang}'
        ent_key = f'ents_{shuffle_lang}'
        shuffled_ents = []
        shuffled_indexes = [i for i in range(len(sample[ent_key]))]
        random.shuffle(shuffled_indexes)
        sample.update({f'idx_{shuffle_lang}': shuffled_indexes})
        
        # get non-entity positions and strings from the original text
        blanks = []
        for i in range(len(sample[ent_key])-1):
            end_prev = sample[ent_key][i][1]
            start_foll = sample[ent_key][i+1][0]
            blanks.append([end_prev, start_foll, sample[text_key][end_prev:start_foll]])
            
        ent_start = 0
        shuffled_text = ''
        for new, idx in enumerate(sample[f'idx_{shuffle_lang}']):
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
        sample.update({
            text_key: shuffled_text,
            ent_key: shuffled_ents,
        })
        if verbose:
            print_list = []
            for i in range(len(sample[f'idx_{shuffle_lang}'])):
                row = []
                for l in sample['sample_langs']:
                    row.append([[sample[f'idx_{l}'].index(i)], sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)] + [sample[f'text_{l}'][sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)][0]:sample[f'ents_{l}'][sample[f'idx_{l}'].index(i)][1]]]])
                print_list.append(row)
            print(pd.DataFrame(print_list))
            print(sample['idx_en'])
            print(sample['idx_it'])

        return sample

class XLWADataset(DatasetDict):
    def __init__(self,
                data_path,
                tokenizer,
                languages = ['it'],
                src_context = True,
                n_rows = None,
                batch_size = None,
                splits = ['train', 'dev', 'test']
                    ):
        '''
        Prepares data from the XL-WA dataset based on the wanted languages,
        tokenizer, number of rows and whether
        we want source context or not surrounding the query word.

        Args:
            data_path (`str` or `os.PathLike`):
                Path to the main XL-WA directory of the format "/path/to/XL-WA".
            tokenizer (`transformers.models.*`):
                Instance of transformers tokenizer.
            languages (`List[str]`, defaults to `['it']`):
                Language code of the target languages to include in the dataset.
            src_context (`bool`, defaults to `True`):
                if `True`, the query is the whole source sentence and the word being aligned
                is surrounded with "•" characters.
            n_rows (`int`, defaults to `None`):
                Defines how many lines are going to be kept in the dataset. Normally used to speed up testing.
            batch_size (`int`, defaults to `None`):
                Size of the train and dev batches.
                If None, the whole dataset is tokenized to the token length of the longest sample.
            splits (`List[str]`, defaults to `['train', 'dev', 'test']`):
                List of names of the splits to include in the dataset.
        '''
        self.name = f"xlwa_en-{'-'.join(languages)}"
        self.tokenizer = tokenizer
        self.languages = languages
        self.src_context = src_context
        self.n_rows = n_rows
        ds = {key: [] for key in splits}
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
                    lambda row: self.convert_giza(row[0], row[1], row[2]), axis=1)
                unpacked_lines += df[:n_rows].to_dict(orient='records')
            progbar = tqdm(unpacked_lines, total = len(unpacked_lines))
            progbar.set_description(f'Creating samples for {split} split...')
            for line in progbar:
                ds[split] += self.create_samples_xlwa(line)
            self[split] = Dataset.from_list(ds[split])

        dataset = self.map(
            lambda sample: self.qa_tokenize(sample, self.tokenizer),
            batched = True,
            batch_size = batch_size,
            )
        
        for key in dataset.keys():
            self[key] = dataset[key]
        
        self.set_format('torch', columns = ['input_ids',
                                    'token_type_ids',
                                    'attention_mask',
                                    'start_positions',
                                    'end_positions']
                                    )

    def create_samples_xlwa(self, sample) -> List[dict]:
        '''
        Creates one sample per word in the source sentence of the sample.
        '''
        sample_list = []
        for pair in sample['pairs']:
            new_sample = {}

            if self.src_context:
                new_sample['query'] = sample['src'][:pair['src_start']] +\
                            '• ' + sample['src'][pair['src_start']:pair['src_end']] + ' •' +\
                            sample['src'][pair['src_end']:]
            else:
                new_sample['query'] = sample['src'][pair['src_start']:pair['src_end']]

            new_sample['context'] = sample['tgt']
            new_sample['answer'] = sample['tgt'][pair['tgt_start']:pair['tgt_end']]
            new_sample['answer_start'] = pair['tgt_start']
            new_sample['answer_end'] = pair['tgt_end']
            query_enc = self.tokenizer(new_sample['query'])
            l = len(query_enc['input_ids']) 
            context_enc = self.tokenizer(new_sample['context'])

            # start_positions and end_positions are token positions
            new_sample['start_positions'] = context_enc.char_to_token(
                new_sample['answer_start']) - 1 + l
            new_sample['end_positions'] = context_enc.char_to_token(
                new_sample['answer_end']-1) + l
            sample_list.append(new_sample)
        return sample_list

    def convert_giza(self,  src_sentence, tgt_sentence, alignments) -> dict:
        '''
        Converts GIZA-style '0-1' string alignments to dicts like:
            {
                'src_start': src_span[0],
                'src_end': src_span[1],
                'tgt_start': tgt_span[0],
                'tgt_end': tgt_span[1],
            }
        '''
        src_spans = self.calculate_spans(src_sentence)
        tgt_spans = self.calculate_spans(tgt_sentence)
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

    def calculate_spans(self, sentence):
        spans = []
        start = 0

        for word in sentence.split():
            end = start + len(word)
            spans.append((start, end))
            start = end + 1
        return spans
    
    @staticmethod
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

def data_loader(dataset, batch_size, n_rows = None):
    '''
    Args:
        dataset (`DatasetDict`):
            Dataset to load.
        batch_size (`int`):
            Size of the training/dev/test batches.
        n_rows (`int`, defaults to `None`):
            Defines how many dataset lines (not DataLoader batches!) are going to be kept in the dataset. Normally used to speed up testing.
    '''
    dl_dict = {}
    for split in dataset.keys():
        dl_dict[split] = DataLoader(dataset[split].select(
                            range(len(dataset[split]))[:n_rows]),
                            batch_size = batch_size,
                            # shuffle = True
                            )
    return dl_dict

def push_model_card(repo_id, model_description = '', results = '', language = 'en', template_path = None):
    repo_id = repo_id
    card_data = ModelCardData(language=language,
                              license='mit',
                              library_name='pytorch',
                              metrics='exact_match'
                              )
    card = ModelCard.from_template(
        card_data,
        model_description = model_description,
        results = results,
        developers = "Paolo Gajo",
        template_path=template_path
    )
    card.push_to_hub(repo_id)
    return repo_id

def push_dataset_card(repo_id, *, dataset_summary, model_metrics = '', template_path = None):
    DatasetCard.from_template(DatasetCardData(),
                            dataset_summary=dataset_summary,
                            model_metrics = '',
                            template_path = template_path,
                            ).push_to_hub(repo_id)

def save_local_model(model_dir, model, tokenizer):
    print(f'Saving model to directory: {model_dir}')
    # Save the model and tokenizer in the local repository
    if hasattr(model, 'module'):
        model.module.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

def print_ents_tasteset_sample(path):
    df_list = []
    with open(path, encoding='utf8') as f:
        data = json.load(f)
    for sample in data['annotations']:
        row = []
        for i in range(len(sample['ents_it'])):
            row.append(sample['text_it'][sample['ents_it'][i][0]:sample['ents_it'][i][1]])
        df_list.append(row)
        # for i in range(len(sample['ents_it'])):
        #     print(i, sample['text_it'][sample['ents_it'][i][0]:sample['ents_it'][i][1]], sep=' -- ', end = '|')
    for line in df_list:
        print(line)

def token_span_to_char_indexes(input, start_index_token, end_index_token, sample, tokenizer, target_text_key = 'text_it'):
    tokens = input['input_ids'].squeeze()
    token_span = tokens[start_index_token:end_index_token]
    token_based_prediction = tokenizer.decode(token_span)
    start = input.token_to_chars(start_index_token)[0]
    if 'deberta_v2' in str(tokenizer.__class__).split('.'):
        end = start + len(token_based_prediction)
        char_span_prediction = sample[target_text_key][start:end]
        if char_span_prediction[0] == ' ':
            start += 1
            end += 1
    elif 'bert' in str(tokenizer.__class__).split('.'):
        # end_span = input.token_to_chars(end_index_token)
        # end = end_span[1]
        end = start + len(token_based_prediction)
    char_span_prediction = sample[target_text_key][start:end]
    # print('char_span_prediction', [char_span_prediction])
    return start, end

def tasteset_to_label_studio(annotation_list, model_name):
    tasks = []
    for recipe in annotation_list:
        predictions = []
        results = []
        languages = ['en', 'it']
        entry = {}
        for language in languages:
            for entity in recipe[f'ents_{language}']:
                results.append({
                    'from_name': f'label_{language}',
                    'to_name': f'text_{language}_ref',
                    'type': 'labels',
                    'value': {
                        'start': entity[0],
                        'end': entity[1],
                        'labels': [entity[2]]}
                        })
            entry[f'text_{language}'] = recipe[f'text_{language}']
        predictions.append({'model_version': model_name, 'result': results})
        tasks.append({
            'data': entry,
            'predictions': predictions,
        })
    return tasks

# def label_studio_to_tasteset(data, src_lang = 'it'):
#     formatted_data = []
#     languages = [key.split('_')[-1] for key in data[0]['data'].keys() if '_' in key]
#     tgt_langs = [lang for lang in languages if lang != src_lang]
#     for line in data:
#         tmp_dict = {}
#         for tgt_lang in tgt_langs:
#             tmp_dict[f'text_{tgt_lang}'] = line['data'][f'ingredients_{tgt_lang}']
#         tmp_dict[f'text_{src_lang}'] = line['data'][f'ingredients_{src_lang}']
#         tmp_dict[f'ents_{src_lang}'] = []
#         for ent in line['annotations'][0]['result']:
#             tmp_dict[f'ents_{src_lang}'].append([ent['value']['start'], ent['value']['end'], ent['value']['labels'][0]])
#         tmp_dict[f'ents_{src_lang}'].sort()
#         formatted_data.append(tmp_dict)
#     return {'annotations': formatted_data}

