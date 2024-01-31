from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_printoptions(linewidth=10000)
model_name = 'pgajo/mdeberta_tasteset_U0_S1_E10_DEV98.0_DROP1'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import json
from tqdm.auto import tqdm
bilingual_path = '/home/pgajo/working/food/data/TASTEset/data/formatted/TASTEset_sep_format_en-it_unaligned.json'
with open(bilingual_path, encoding='utf8') as f:
    data = json.load(f)

recipe_list = data['annotations']

for idx, recipe in tqdm(enumerate(recipe_list), total=len(recipe_list)):
    for i, entity in enumerate(recipe['entities_en']):
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
        
        # start_scores = outputs.start_logits
        # end_scores = outputs.end_logits
        # start_index_token = int(torch.argmax(start_scores))
        # end_index_token = int(torch.argmax(end_scores))
        
        # print('start_index_token', start_index_token)
        # print('end_index_token', end_index_token)
        # print('len(input_ids)', len(input_ids))
        if start_index_token >= len(input_ids) - 1 or end_index_token >= len(input_ids) - 1:
            continue
        # print('encoding:', input_ids)
        decoded_input = tokenizer.decode(input_ids)
        # print('decoded:', decoded_input)
        # for j, id in enumerate(input_ids):
        #     print(j, int(id), tokenizer.decode([id]), end='\t\t')
        # print()
        # print('prediction_tokens:', input['input_ids'].squeeze()[start_index_token:end_index_token])
        # print('prediction:', tokenizer.decode(input['input_ids'].squeeze()[start_index_token:end_index_token]))
        # print('gold:', [recipe['text_en'][entity[0]:entity[1]]])
        char_span_start = input.token_to_chars(start_index_token)
        # print('char_span_start', char_span_start)
        char_span_end = input.token_to_chars(end_index_token-1)
        char_span_prediction = recipe['text_it'][char_span_start[0]:char_span_end[1]]
        char_span_prediction_splitjoined = ''.join(char_span_prediction.split()).replace('#', '')
        # print('char_span_prediction', )
        # print('char_span_end', char_span_end)
        char_span = (char_span_start[0], char_span_end[1])
        # print('char_span', char_span)
        # print('char_span[0]', char_span[0])
        # print('char_span[1]', char_span[1])
        if not char_span[0] > char_span[1]:
            recipe['entities_it'].append([char_span[0], char_span[1], recipe['entities_en'][i][2]])
        
        token_based_prediction = tokenizer.decode(input['input_ids'].squeeze()[start_index_token:end_index_token])
        token_based_prediction_splitjoined = ''.join(token_based_prediction.split()).replace('#', '')
        gold = recipe['text_en'][entity[0]:entity[1]]