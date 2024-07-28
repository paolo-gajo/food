from utils_food import EntityShifter, mappings
import json

json_path = './data/mycolombianrecipes/mycolombianrecipes_ls.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

shifter = EntityShifter(languages=['en', 'es'], mappings=mappings)


data_shifted = []
for sample in data:
    sample_shifted = shifter.sub_shift(sample, verbose=True)
    data_shifted.append(sample_shifted)

json_path_out = json_path.replace('.json', '_shifted.json')

with open(json_path_out, 'w', encoding='utf8') as f:
    json.dump(data_shifted, f, ensure_ascii = False)

