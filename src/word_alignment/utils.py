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
from huggingface_hub import login, ModelCard, ModelCardData, HfApi, CommitOperationAdd
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
        self.model = model
        self.best_model = model

    def evaluate(self, split, epoch, eval_metric = 'dev'):
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
                if hasattr(self.model, 'module'):
                    self.best_model = self.model.module
                else:
                    self.best_model = self.model
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
            metrics_save_name = os.path.join(path, dt_string + self.best_model.config._name_or_path + exact_dev_best_suffix)
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

class TASTEset(DatasetDict):
    def __init__(self,
        input_data,
        tokenizer_name = 'bert-base-multilingual-cased',
        src_lang = 'en',
        shuffle_languages = ['it'],
        drop_duplicates = True,
        src_context = True,
        shuffled_size = 0,
        unshuffled_size = 1,
        dev_size = 0.2,
        batch_size = None,
        debug_dump = False,
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
            shuffle_languages (`str` or `List[str]`):
                String of a single 2-character language code (ISO 639-1),
                string of multiple space-separated language codes, or list of language codes.
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
            
        Returns:
            `TASTEset`: The raw dataset in the Hugging Face dictionary format.
        '''
        self.name = 'tasteset'
        self.input_data = input_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.shuffle_languages = shuffle_languages
        if isinstance(self.shuffle_languages, str):
            self.shuffle_languages = self.shuffle_languages.split()
        self.src_lang = src_lang
        self.drop_duplicates = drop_duplicates
        self.src_context = src_context
        self.shuffled_size = shuffled_size
        self.unshuffled_size = unshuffled_size
        self.dev_size = dev_size
        self.samples = []
        self.shuffled_samples = []
        self.unshuffled_samples = []
        self.debug_dump = debug_dump

        self.raw_data = self.prep_data()

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
        
    @classmethod
    def from_json(cls,
                  json_path,
                  n_rows = None,
                  **kwargs
                  ) -> "TASTEset":
        input_data = json.load(open(json_path))['annotations'][:n_rows]
        return cls(
            input_data,
            **kwargs
        )

    def prep_data(self) -> 'DatasetDict':
        if self.unshuffled_size:
            self.unshuffled_samples = self.generate_samples(SampleList(self.extend(self.input_data, self.unshuffled_size), shuffle = False))
        if self.shuffled_size:
            self.shuffled_samples = self.generate_samples(SampleList(self.extend(self.input_data, self.shuffled_size), shuffle = True))
        df = pd.concat([pd.DataFrame(self.unshuffled_samples),
                        pd.DataFrame(self.shuffled_samples)])
        if self.drop_duplicates:
            df = df.drop_duplicates(['answer'])
        df_list = df.to_dict(orient = 'records')
        if self.debug_dump:
            json.dump(df_list, open('debug_dump.json', 'w'), ensure_ascii=False)
        ds_dict = Dataset.from_list(df_list).train_test_split(test_size = self.dev_size)
        self['train'] = ds_dict['train']
        ds_dict['dev'] = ds_dict.pop('test')
        self['dev'] = ds_dict['dev']
        return ds_dict

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
        progbar = tqdm(sample_list.samples, total = len(sample_list.samples))
        progbar.set_description(f'Creating samples ({shuffle_desc})...')
        list_buffer = []
        for line in progbar:
            line_buffer = copy.deepcopy(line)
            list_buffer += self.samples_from_line(line_buffer, sample_list.shuffle)
        return list_buffer

    def samples_from_line(self,
                        sample,
                        shuffle,
                        ) -> List[dict]:
        '''
        Creates one sample per word in the source sentence of the sample.
        '''
        sample = self.assign_indexes_to_entities(sample)
        tgt_lang = ''.join([lang for lang in sample['sample_langs'] if lang != self.src_lang])
        sample_list = []
        if shuffle:
            for shuffle_lang in self.shuffle_languages:
                sample = self.shuffle_entities(sample, shuffle_lang)
        for i in range(sample['num_ents']):
            new_sample = {}

            if self.src_context:
                new_sample['query'] = sample[f'text_{self.src_lang}'][:sample[f'ents_{self.src_lang}'][i][0]] +\
                            '• ' + sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][i][0]:sample[f'ents_{self.src_lang}'][i][1]] + ' •' +\
                            sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][i][1]:]
            else:
                new_sample['query'] = sample[f'text_{self.src_lang}'][sample[f'ents_{self.src_lang}'][0]:sample[f'ents_{self.src_lang}'][1]]

            new_sample['context'] = sample[f'text_{tgt_lang}']
            new_sample['answer_start'] = sample[f'ents_{tgt_lang}'][sample[f'idx_{tgt_lang}'].index(i)][0]
            new_sample['answer_end'] = sample[f'ents_{tgt_lang}'][sample[f'idx_{tgt_lang}'].index(i)][1]
            new_sample['answer'] = sample[f'text_{tgt_lang}'][new_sample['answer_start']:new_sample['answer_end']]
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

    @staticmethod
    def assign_indexes_to_entities(sample) -> dict:
        sample_langs = list(set([key.split('_')[-1] for key in sample.keys() if '_' in key]))
        original_indexes = [i for i in range(len(sample[f'ents_{sample_langs[0]}']))]
        for l in sample_langs:
            sample.update({f'idx_{l}': original_indexes,
                        'sample_langs': sample_langs,
                        'num_ents': len(original_indexes)})
        return sample

    @staticmethod
    def shuffle_entities(sample, shuffle_lang, verbose = False) -> dict:
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
        self.name = 'xlwa'
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

def upload_folder_hf(model_dir,
                save_name = None,
                user = 'pgajo',
                 ):
    repo_id = f"{user}/{save_name}"
    print(f'Pushing model to repo: {repo_id}')
    # Initialize a new repository
    api = HfApi()
    api.create_repo(repo_id, token=os.environ['HF_WRITE_TOKEN'])
    api.upload_folder(repo_id=repo_id,
                      folder_path=model_dir,
                      token=os.environ['HF_WRITE_TOKEN']
                      )
    return repo_id

def push_model(model,
               save_name = None,
               user = 'pgajo',
               ):
    # save best model
    if hasattr(model, 'module'):
        model = model.module
    login(token=os.environ['HF_WRITE_TOKEN'])
    if model_name is None:
        model_name = model.config._name_or_path
    repo_id = f"{user}/{save_name}"
    model.push_to_hub(repo_id)
    return repo_id

def push_card(repo_id, model_name, model_description = '', language = 'en'):
    repo_id = repo_id
    card_data = ModelCardData(language=language,
                              license='mit',
                              library_name='pytorch')
    card = ModelCard.from_template(
        card_data,
        model_id = model_name.split('/')[-1],
        model_description = model_description,
        developers = "Paolo Gajo",
    )
    card.push_to_hub(repo_id)
    return repo_id

def save_local_model(model_dir, model, tokenizer):
    print(f'Saving model to directory: {model_dir}')
    # Save the model and tokenizer in the local repository
    if hasattr(model, 'module'):
        model.module.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)