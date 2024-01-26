import os
from utils import prep_tasteset, qa_tokenize
from transformers import AutoTokenizer

def build_tasteset(data_path,
                   shuffle_ents=False,
                   shuffle_languages=['it'],
                   src_lang = 'en',
                ):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    dataset = prep_tasteset(data_path,
                        tokenizer,
                        shuffle_ents=shuffle_ents,
                        shuffle_languages=shuffle_languages,
                        src_lang = src_lang,
                        ).shuffle(seed=42)

    dataset = dataset.map(lambda sample: qa_tokenize(sample, tokenizer),
                        batched=True,
                        batch_size=None,
                        )

    dataset.set_format('torch', columns=['input_ids',
                                        'token_type_ids',
                                        'attention_mask',
                                        'start_positions',
                                        'end_positions'])        
    return dataset

def main():
    json_path = '/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TASTE_en-it_DEEPL.json'
    dir_path = os.path.dirname(json_path)
    lang_list = [
      # 'ru',
      # 'nl',
      'it',
      # 'pt',
      # 'et',
      # 'es',
      # 'hu',
      # 'da',
      # 'bg',
      # 'sl',
      ]
    
    dataset = build_tasteset(json_path,
                             shuffle_ents=True,
                             shuffle_languages='it',
                             src_lang = 'en',
                             )
    lang_id = '-'.join(lang_list)
    output_path = os.path.join(dir_path, f'.ready/{lang_id}')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # print(dataset)
    # print(dataset['train'])
    # print(dataset['train'][0])
    
    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    main()