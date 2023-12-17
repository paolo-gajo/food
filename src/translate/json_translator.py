from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import os
import json
from typing import List, Dict
import argparse

from os import environ

from google.cloud import translate

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0]

def translate_marianmt(text, tokenizer, model, device):
    batch = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(**batch, max_new_tokens=1024)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def load_json(json_path):
    with open(json_path, encoding='utf8') as json_file:
        data = json.load(json_file)
    return data

def translate_list_of_texts(texts: List[str], tokenizer, model, device, verbose=False) -> List[str]:
    trg_texts = []

    # Process each recipe's annotations
    for text in tqdm(texts, desc="Translating"):
        src_text = text
        trg_text = translate_marianmt(src_text, tokenizer, model, device)
        trg_texts.append(trg_text)

        if verbose:
            print("SRC:", src_text)
            print("TRG:", trg_text)
    
    return trg_texts

def main():
    parser = argparse.ArgumentParser(description="Translate text using a pre-trained model.")
    parser.add_argument("--model", default='Helsinki-NLP/opus-mt-tc-big-en-it', help="The name or path of the pre-trained model to use")
    parser.add_argument("--json_path", default='/home/pgajo/working/food/data/TASTEset/data/TASTEset_semicolon.json', help="The path to the JSON file with input data")
    parser.add_argument("--num_texts", type=int, default=-1, help="The number of texts to translate")
    args = parser.parse_args()
    model = args.model  # Get the model name from the command line argument

    tokenizer = AutoTokenizer.from_pretrained(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    json_path = args.json_path

    data_src = load_json(json_path)

    if args.num_texts == -1:
        src_texts = [el['text'] for el in data_src['annotations']]
    else:
        src_texts = [el['text'] for el in data_src['annotations'][:args.num_texts]]
    
    with open(json_path.replace('.json', '_srctest.txt'), 'w', encoding='utf8') as f:
        for el in src_texts:
            f.write(el+"\n")
    
    trg_texts = (translate_marianmt("\n".join(src_texts), "it")).split("\n")

    with open(json_path.replace('.json', '_trgtest.txt'), 'w', encoding='utf8') as f:
        for el in trg_texts:
            f.write(el+"\n")

    # PROJECT_ID = environ.get("PROJECT_ID", "")
    # assert PROJECT_ID
    # PARENT = f"projects/{PROJECT_ID}"

if __name__ == '__main__':
    main()  
