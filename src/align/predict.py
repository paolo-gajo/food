from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import token_span_to_char_indexes, TASTEset
import torch
torch.set_printoptions(linewidth=10000)
# from icecream import ic
import json
from tqdm.auto import tqdm
# import argparse
# import sys
# sys.path.append('/home/pgajo/food/src')
# from utils_food import TASTEset

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model_name = '/home/pgajo/food/models/alignment/en-it-es/best/mbert_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.1_en-it-es_ME3_2024-04-24-00-24-38_TEST_GZ=49.46'
    # model_name = '/home/pgajo/food/models/alignment/en-it-es/best/mdeberta_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-24-06-08-49_TEST_GZ=61.87'
    
    # model_name = '/home/pgajo/food/models/alignment/en-it-es/best/mbert_xlwa_en-it-es_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-24-03-17-41_TEST_MCR=70.61'
    model_name = '/home/pgajo/food/models/alignment/en-it-es/best/mdeberta-v3-base_EW-TT-MT_multi_ctx_P0.2_en-it-es_ME3_2024-04-23-21-21-18_TEST_MCR=75.56'

    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    json_path_unaligned = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_DEEPL_unaligned.json'
    with open(json_path_unaligned, 'r', encoding='utf8') as f:
        data = json.load(f)

    lang_src = 'en'
    lang_tgt = 'es'
    field_name = 'ingredients'

    progbar = tqdm(enumerate(data), total=len(data))
    num_ents = 0
    num_errors = 0
    progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors}')
    for idx, recipe in progbar:
        for i, entity in enumerate(recipe[f'ents_{lang_src}']):
            num_ents += 1
            query = recipe[f'{field_name}_{lang_src}'][:entity[0]] + '• ' + recipe[f'{field_name}_{lang_src}'][entity[0]:entity[1]] + ' •' + recipe[f'{field_name}_{lang_src}'][entity[1]:]
            context = recipe[f'{field_name}_{lang_tgt}']
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
                start, end = token_span_to_char_indexes(input, start_index_token, end_index_token, recipe, tokenizer, field_name=field_name, lang=lang_tgt)
                # ic(recipe['{field_name}_en'][entity[0]:entity[1]])
                # ic(recipe[f'{field_name}_{lang_tgt}'][start:end])
                recipe[f'ents_{lang_tgt}'].append([start, end, entity[2]])
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

    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(filename)

    ls_data = TASTEset.tasteset_to_label_studio(data, model_name)

    ls_filename = filename.replace('.json', '_ls.json')
    with open(ls_filename, 'w', encoding='utf8') as f:
        json.dump(ls_data, f, ensure_ascii=False)

if __name__ == '__main__':
    main()