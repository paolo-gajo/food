import json

with open('/home/pgajo/working/food/TASTEset/data/TASTEset_semicolon_translated.json') as f:
    data = json.load(f)

for recipe in data['annotations']:
    for entity in recipe['entities']:
        start = entity[0]
        end = entity[1]
        quantity = entity[2]
        print(recipe['text'][start:end], start, end, quantity)