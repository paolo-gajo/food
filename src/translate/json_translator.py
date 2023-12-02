import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import os

# Pseudo code:
    # 1. look at each element in 'entities', e.g. [0, 1, 'QUANTITY'], and find the equivalent to the positions in 'text'
    # 2. translate that equivalent and append it to a new list
    # 3. get the difference in length between src and trg and add that to all starts and ends in each following element of entities_trg

def translate_marianmt(text, tokenizer, model, device):
    """
    Translate a piece of text using the MarianMT model.

    Parameters:
    - text: The source text to translate.
    - tokenizer: The tokenizer for the model.
    - model: The translation model.
    - device: The device to run the model on.

    Returns:
    - The translated text.
    """
    batch = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(**batch, max_new_tokens=1024)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def has_alphanumeric(s):
    return any(char.isalnum() for char in s)

def translate_TASTEset(json_path, tokenizer, model, device, verbose=False):
    """
    Translate the annotations within a TASTEset JSON file.

    Parameters:
    - json_path: Path to the TASTEset JSON file.
    - tokenizer: The tokenizer for the model.
    - model: The translation model.
    - device: The device to run the model on.
    - verbose: If set to True, prints additional information.
    """

    with open(json_path, encoding='utf8') as json_file:
        data = json.load(json_file)

    new_data = {'annotations': data['annotations'][:5]}

    # Process each recipe's annotations
    for recipe in tqdm(new_data['annotations'], desc="Translating"):
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
    print('new_data:', new_data)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    output_path = json_path.replace('.json', '_translated.json')
    output_path = os.path.join(os.path.dirname(output_path), time_string + os.path.basename(output_path))
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, indent=4, ensure_ascii=False)
    print(f"Translated data saved to {output_path}")

def main():
    """
    Main function to initialize the model and tokenizer, then translate a given JSON file.
    """
    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-it'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    json_path = '/home/pgajo/working/food/data/TASTEset/data/TASTEset_semicolon.json'
    translate_TASTEset(json_path, tokenizer, model, device, verbose=2)

if __name__ == '__main__':
    main()  
