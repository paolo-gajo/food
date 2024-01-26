import json
from utils import shuffle_entities
with open('/home/pgajo/working/food/data/TASTEset/data/entity-wise/EW-TASTE_en-it_DEEPL.json') as f:
    data = json.load(f)
recipe_list = data['annotations']
print(len(recipe_list))
sample = recipe_list[5]

shuffled_sample = shuffle_entities(sample, 'it', verbose=True)
# print(shuffled_sample)