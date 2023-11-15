import spacy
import json
import sys
sys.path.append('/home/pgajo/working/food/TASTEset/src')
from utils import prepare_data, ENTITIES

recipes, entities = prepare_data("/home/pgajo/working/food/TASTEset/data/TASTEset_semicolon.csv")

annotations = [{'text': ' '.join(recipe.splitlines()), 'entities': ents} for recipe, ents in zip(recipes, entities)]
# print(annotations[:3])
training_data = {'classes': ENTITIES, 'annotations': annotations}
# print(training_data['annotations'])

import json
with open('/home/pgajo/working/food/TASTEset/data/TASTEset_semicolon.json', 'w') as fp:
    json.dump(training_data, fp)