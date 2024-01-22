import json
with open('/home/pgajo/working/food/data/TASTEset/data/entity-wise/EW-TASTE_en-it_DEEPL.json') as f:
    data = json.load(f)
recipe_list = data['annotations']
print(len(recipe_list))
import copy
sample = recipe_list[5]
import random

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

shuffled_sample = shuffle_entities(sample, 'en')
print(shuffled_sample)

for i in range(len(shuffled_sample['entities_en'])):
    print(i, shuffled_sample['entities_en'][i], shuffled_sample['text_en'][shuffled_sample['entities_en'][i][0]:shuffled_sample['entities_en'][i][1]])