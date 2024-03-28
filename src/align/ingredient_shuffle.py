import json

with open('/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TT-PE.json', 'r', encoding='utf8') as f:
    data = json.load(f)

import random
import pandas as pd
import re
import copy 

def shuffle_entities_ingredient(sample, shuffle_lang, verbose = False) -> dict:
    '''
    Shuffles entities and text for a sample.
    Only shuffles, so the source and target entities need 
    to be linked somehow after shuffling one language
    (usually the target language)!
    '''
    
    text_key = f'text_{shuffle_lang}'
    ent_key = f'ents_{shuffle_lang}'
    sample_text = sample[text_key]
    semicolon_positions = [0] + [match.end() for match in re.finditer('(?<!\s);(?!\s)', sample_text)] + [len(sample_text)]

    shuffled_list = []
    shuffled_text = ''

    for pos in range(1, len(semicolon_positions)):
        scope_start = semicolon_positions[pos - 1]
        scope_end = semicolon_positions[pos]
        ingr = []
        for ent in sample[ent_key]:
            if ent[1] <= scope_end and ent[0] >= scope_start:
                ingr.append(copy.deepcopy(ent))

        shuffled_ents = []
        shuffled_indexes = [i for i in range(len(ingr))]
        random.shuffle(shuffled_indexes)
        # sample.update({f'idx_{shuffle_lang}': shuffled_indexes})

        # get non-entity positions and strings from the original text
        blanks_ingr = []
        for i in range(1, len(ingr)):
            end_prev = ingr[i-1][1]
            start_foll = ingr[i][0]
            blanks_ingr.append([end_prev, start_foll, sample_text[end_prev:start_foll]])

        ent_start = ingr[0][0]
        shuffled_text_ingr = ''
        for idx in shuffled_indexes:
            tmp_ent = ingr[idx]
            text_tmp_ent = sample_text[ingr[idx][0]:ingr[idx][1]]
            
            len_text = len((text_tmp_ent))
            tmp_ent[0] = ent_start
            tmp_ent[1] = tmp_ent[0] + len_text
            tmp_ent[2] = ingr[idx][2]
            shuffled_ents.append(tmp_ent)

            if len(blanks_ingr) > 0:
                next_blank = blanks_ingr.pop(0)
                ent_start += len((text_tmp_ent)) + next_blank[1] - next_blank[0]
            else:
                next_blank = ['', '', '']
                pass

            shuffled_text_ingr += text_tmp_ent + next_blank[2]
        shuffled_list += shuffled_ents
        shuffled_text += shuffled_text_ingr
        shuffled_text = shuffled_text + sample_text[len(shuffled_text):scope_end]
        print(shuffled_text)
    
    sample.update({
        text_key: shuffled_text,
        ent_key: shuffled_list,
    })
    return sample

sample = data['annotations'][243]
print(sample)
# for key in sample.keys():
#     print(f"{key}:\t", sample[key])
# print('-----------------')

# for sample in data['annotations'][3:4]:
sample_shuffled = shuffle_entities_ingredient(sample, 'it')
for key in sample_shuffled.keys():
    print(f"{key}:\t", sample_shuffled[key])
print('-----------------')

# for ent in sample_shuffled['ents_it']:
#     print(sample_shuffled['text_it'][ent[0]:ent[1]])