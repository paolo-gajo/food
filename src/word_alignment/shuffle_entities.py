import json
with open('/home/pgajo/working/food/data/TASTEset/data/entity-wise/EW-TASTE_en-it_DEEPL.json') as f:
    data = json.load(f)
recipe_list = data['annotations']
print(len(recipe_list))
import copy
sample = copy.deepcopy(recipe_list[5])
import random

def shuffle_entities(sample):
    shuffled_sample = []
    sample_old = copy.deepcopy(sample)
    print(sample_old['entities_en'])
    dict_ent = {}
    # for i, el in enumerate(sample_old['entities_en']):
    #     # print(i, el)
    #     if el[2] not in dict_ent.keys():
    #         dict_ent[el[2]] = 1
    #     else:
    #         dict_ent[el[2]] += 1
    # dict_ent = dict(sorted(dict_ent.items()))
    # print(dict_ent)
    indexes = [i for i in range(len(sample_old['entities_en']))]
    random.shuffle(indexes)
    print(indexes)
    blanks = []
    for i in range(len(sample['entities_en'])-1):
        end_prev = sample['entities_en'][i][1]
        start_foll = sample['entities_en'][i+1][0]
        blanks.append([end_prev, start_foll, 'BLANK'])
    ent_start = 0
    for new, old in enumerate(indexes):
        tmp_ent_en = copy.deepcopy(sample_old['entities_en'][old])
        text_tmp_ent_en = sample_old['text_en'][sample_old['entities_en'][old][0]:sample_old['entities_en'][old][1]]
        tmp_ent_en[0] = ent_start
        tmp_ent_en[1] = tmp_ent_en[0] + len((text_tmp_ent_en))
        tmp_ent_en[2] = sample_old['entities_en'][old][2]
        
        if len(blanks) > 0:
            next_blank = blanks.pop(0)
            ent_start += len((text_tmp_ent_en)) + next_blank[1] - next_blank[0]
        else:
            pass

        shuffled_sample.append(tmp_ent_en)
    return shuffled_sample

shuffled_sample = shuffle_entities(sample)
dict_ent = {}
# for i, el in enumerate(shuffled_sample):
#     # print(i, el)
#     if el[2] not in dict_ent.keys():
#         dict_ent[el[2]] = 1
#     else:
#         dict_ent[el[2]] += 1
# dict_ent = dict(sorted(dict_ent.items()))
# print(dict_ent)