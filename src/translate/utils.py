import json
from typing import List
from tqdm.auto import tqdm

def translate_TASTEset_old(data_trg: Dict, tokenizer, model, device, verbose=False):
    """
    Translate the annotations within a TASTEset JSON file.

    Parameters:
    - json_path: Path to the TASTEset JSON file.
    - tokenizer: The tokenizer for the model.
    - model: The translation model.
    - device: The device to run the model on.
    - verbose: If set to True, prints additional information.
    """

    # Process each recipe's annotations
    for recipe in tqdm(data_trg['annotations'], desc="Translating"):
        new_text = ''
        entities_trg = []
        shift = 0
        for entity in recipe['entities']:
            print('-------------------')
            start_src, end_src, entity_type = entity
            entity_text_src = recipe['text'][start_src:end_src]
            if verbose > 1:
                print('source:', entity_text_src, start_src, end_src)
            entity_text_trg = translate_marianmt(entity_text_src, tokenizer, model, device)

            start_trg = len(new_text)

            shift += len(entity_text_trg) - len(entity_text_src)

            new_text += entity_text_trg

            end_trg = len(new_text)

            if verbose > 1:
                print('target:', new_text[start_trg:end_trg], start_trg, end_trg)

            if recipe['text'][end_src:end_src+1] == ';':
                end_char_trg = recipe['text'][end_src:end_src+1]
            else:
                # look for the next ; in the source text
                # if there are any alphanumeric characters before it
                # then end_char = ' ', else end_char = '; '
                end_char_src = recipe['text'][end_src:].find(';')
                if end_char_src == -1:
                    # -1 means no ; was found
                    end_char_trg = recipe['text'][end_src:end_src+1]
                else:
                    end_char_src += end_src
                    print("recipe['text'][end_src:end_char_src]", recipe['text'][end_src:end_char_src])
                    if has_alphanumeric(recipe['text'][end_src:end_char_src]):
                        end_char_trg = recipe['text'][end_src:end_src+1]
                    else:
                        end_char_trg = '; '
                    
            print('end_char_trg: "', end_char_trg, '"') if end_char_trg != ' ' else print('end_char_trg: <space>')
            
            new_text += end_char_trg

            entities_trg.append([start_trg, end_trg, entity_type])
            

        if verbose:
            print('original text:', recipe['text'])
            print('new_text:', new_text)
        recipe['text'] = new_text
        recipe['entities'] = entities_trg

    # Output the new data
    print('new_data:', data_trg)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    output_path = json_path.replace('.json', '_translated.json')
    output_path = os.path.join(os.path.dirname(output_path), time_string + os.path.basename(output_path))
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data_trg, outfile, indent=4, ensure_ascii=False)

    print(f"Translated data saved to {output_path}")