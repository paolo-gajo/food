from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils import token_span_to_char_indexes, tasteset_to_label_studio
import torch
torch.set_printoptions(linewidth=10000)
from icecream import ic
import json
from tqdm.auto import tqdm
import argparse

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model_name = '/home/pgajo/food/models/alignment/mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-20-36-43_TEST61.0'
    # model_name = '/home/pgajo/food/models/alignment/EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0/mbert_xlwa_en-it/mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-04-02-17-37-26_TEST53.0'
    model_name = '/home/pgajo/food/models/alignment/EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0/mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-04-03-06-12-17/mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-04-03-06-12-17_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-04-03-10-55-06_TEST63'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    json_path_unaligned = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_unaligned_spaced.json'
    with open(json_path_unaligned, 'r', encoding='utf8') as f:
        data = json.load(f)

    recipe_list = data['annotations']#[:3]
    progbar = tqdm(enumerate(recipe_list), total=len(recipe_list))
    num_ents = 0
    num_errors = 0
    progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors}')
    for idx, recipe in progbar:
        for i, entity in enumerate(recipe['ents_en']):
            num_ents += 1
            query = recipe['text_en'][:entity[0]] + '• ' + recipe['text_en'][entity[0]:entity[1]] + ' •' + recipe['text_en'][entity[1]:]
            context = recipe['text_it']
            input = tokenizer(query, context, return_tensors='pt').to('cuda')
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
                # ic(recipe['text_en'][entity[0]:entity[1]])
                # ic(recipe['text_it'][start:end])
                recipe['ents_it'].append([start, end, entity[2]])
            elif start_index_token < len(tokenizer(query, return_tensors = 'pt')['input_ids'].squeeze()):
                # print('wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf?')
                pass
            else:
                # print('### START TOKEN !< END TOKEN ###')
                num_errors += 1
                pass
            
            progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors} - Err%: {round((num_errors/num_ents)*100, 2)}')

            # print('##########################################')

    filename = f"{json_path_unaligned.replace('.json', '')}_{model_name.split('/')[-1]}.json".replace('unaligned', 'aligned')
    new_data = {'classes': data['classes'], 'annotations': recipe_list}
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(new_data, f, ensure_ascii=False)

    print(filename)

    ls_data = tasteset_to_label_studio(data['annotations'], model_name)

    ls_filename = filename.replace('.json', '_ls.json')
    with open(ls_filename, 'w', encoding='utf8') as f:
        json.dump(ls_data, f, ensure_ascii=False)

if __name__ == '__main__':
    main()