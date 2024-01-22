import os
import pandas as pd
from typing import List, Union
from datasets import Dataset, DatasetDict
import torch
from tqdm.auto import tqdm
import random

class WADataset(DatasetDict):

    def __init__(self,
                 languages: List,
                 data_path: str,
                 n_rows: Union[int, None] = None
                 ) -> List[dict]:
        
        self.splits = ['train', 'dev', 'test']
        self.languages = languages
        self.lang_id = '-'.join(self.languages)
        self.data_path = data_path
        self.n_rows = n_rows

        ds = {key: [] for key in self.splits}
        for split in self.splits:
            ds[split] = []
            for lang in self.languages:
                df = pd.read_csv(os.path.join(self.data_path,
                                                f'{lang}/{split}.tsv'), sep='\t', header=None)
                df.columns = ['src', 'tgt', 'spans']
                df['lang'] = lang
                df['pairs'] = df.apply(
                    lambda row: self.convert_alignments(row[0], row[1], row[2]), axis=1)
                ds[split] += df[:self.n_rows].to_dict(orient='records')
            df_hf = Dataset.from_list(ds[split])
            self[split] = df_hf
    
    @classmethod
    def from_giza_alignments(cls,
                             data_list):
        return cls{
            languages=languages,
            

        }

    def create_samples(self, sample, src_context = True):
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
            query_enc = self.tokenizer(new_sample['query'])
            l = len(query_enc['input_ids']) 
            context_enc = self.tokenizer(new_sample['context'])
            new_sample['answer_start_token'] = context_enc.char_to_token(new_sample['answer_start']) - 1 + l
            new_sample['answer_end_token'] = context_enc.char_to_token(new_sample['answer_end']-1) + l
            sample_list.append(new_sample)
        
        return sample_list
    
    def tokenize_sample(self, sample: Union[str, List]):
        sample.update(self.tokenizer(sample['query'],
                           sample['context'],
                           padding = 'longest',
                           return_tensors = 'pt'
                           ))
        return sample

    def format_and_tokenize(self, tokenizer, shuffle = True):
        self.tokenizer = tokenizer
        for split in self.splits:
            split_samples = []
            progressbar = tqdm(self[split], total = len(self[split]))
            progressbar.set_description('Building word alignment samples from sentences...')
            for line in progressbar:
                split_samples.extend(self.create_samples(line))
            self[split] = Dataset.from_list(split_samples)
            if shuffle:
                self[split] = self[split].shuffle(seed=42)
        return self

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
            converted_alignments.append({
                'src_start': src_span[0],
                'src_end': src_span[1],
                'tgt_start': tgt_span[0],
                'tgt_end': tgt_span[1],
            })
            # converted_alignments.append(((src_span[0],src_span[1]),(tgt_span[0], tgt_span[1])))

        return converted_alignments
    
def shuffle_entities(sample, lang):
    text_key = f'text_{lang}'
    entities_key = f'entities_{lang}'
    shuffled_ents = []
    indexes = [i for i in range(len(sample[entities_key]))]
    random.shuffle(indexes)

    # get non-entity positions and strings from the original text
    blanks = []
    for i in range(len(sample[entities_key])-1):
        end_prev = sample[entities_key][i][1]
        start_foll = sample[entities_key][i+1][0]
        blanks.append([end_prev, start_foll, sample[text_key][end_prev:start_foll]])
        
    ent_start = 0
    shuffled_text = ''
    for new, idx in enumerate(indexes):
        tmp_ent = sample[entities_key][idx]
        text_tmp_ent = sample[text_key][sample[entities_key][idx][0]:sample[entities_key][idx][1]]
        
        len_text = len((text_tmp_ent))
        tmp_ent[0] = ent_start
        tmp_ent[1] = tmp_ent[0] + len_text
        tmp_ent[2] = sample[entities_key][idx][2]
        shuffled_ents.append(tmp_ent)

        if len(blanks) > 0:
            next_blank = blanks.pop(0)
            ent_start += len((text_tmp_ent)) + next_blank[1] - next_blank[0]
        else:
            pass

        shuffled_text += text_tmp_ent + next_blank[2]
    
    shuffled_sample = {
        text_key: shuffled_text,
        entities_key: shuffled_ents
    }

    return shuffled_sample