from utils_food import TASTEset
import json

json_path = './data/mycolombianrecipes/mycolombianrecipes.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

data_ls = TASTEset.tasteset_to_label_studio(data)

json_path_ls = json_path.replace('.json', '_ls.json')

with open(json_path_ls, 'w', encoding='utf8') as f:
    json.dump(data_ls, f, ensure_ascii = False)