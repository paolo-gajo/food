from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import token_span_to_char_indexes
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_printoptions(linewidth=10000)
model_name = 'pgajo/mdeberta_tasteset_U0_S1_E10_DEV98.0_DROP1'
# model_name = 'pgajo/mbert_tasteset_U0_S1_E10_DEV61.0_DROP1'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import json
from tqdm.auto import tqdm
json_path_unaligned = '/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format_en-it_unaligned.json'
with open(json_path_unaligned, encoding='utf8') as f:
    data = json.load(f)

recipe_list = data['annotations'][:3]

for idx, recipe in tqdm(enumerate(recipe_list), total=len(recipe_list)):
    for i, entity in enumerate(recipe['ents_en']):
        input = tokenizer(
            recipe['text_en'][:entity[0]] + '• ' + recipe['text_en'][entity[0]:entity[1]] + ' •' + recipe['text_en'][entity[1]:],
            recipe['text_it'],
            return_tensors='pt',
            ).to('cuda')
        input_ids = input['input_ids'].squeeze()
        with torch.inference_mode():
            outputs = model(**input)

            # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
            for i in range(len(outputs['start_logits'])):
                outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], input['token_type_ids'][i]-10000)
                outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], input['token_type_ids'][i]-10000)
            
        start_index_token = torch.argmax(outputs['start_logits'], dim=1)
        end_index_token = torch.argmax(outputs['end_logits'], dim=1)

        if start_index_token < end_index_token:
            start, end = token_span_to_char_indexes(input, start_index_token, end_index_token, recipe, tokenizer)
            recipe['ents_it'].append([start, end, entity[2]])
        else:
            # print('### START TOKEN !< END TOKEN ###')
            pass

# save aligned dataset to a new json
import os
sw_dir = '/home/pgajo/working/food/data/TASTEset/data/SW-TASTE'
filename = f"{os.path.splitext(os.path.basename(json_path_unaligned))[0]}_{model_name.split('/')[-1]}.json".replace('unaligned', 'aligned')
new_data = {'classes': data['classes'], 'annotations': recipe_list}
# json_path_aligned = os.path.join(sw_dir, filename)
json_path_aligned = 'gz_aligned.json'
with open(json_path_aligned, 'w', encoding='utf8') as f:
    json.dump(new_data, f, ensure_ascii=False)

print(json_path_aligned)