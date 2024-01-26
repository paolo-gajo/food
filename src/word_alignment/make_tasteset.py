import os
from utils import prep_tasteset, qa_tokenize
from transformers import AutoTokenizer

def build_tasteset(data_path,
                shuffle_ents=False,
                shuffle_languages=['it'],
                src_lang = 'en',
                tokenizer_name = 'bert-base-multilingual-cased',
                dev_size=0.2,
                shuffled_size = 1,
                unshuffled_size = 1,
                drop_duplicates = True,
                ):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = prep_tasteset(data_path,
                        tokenizer,
                        shuffle_ents=shuffle_ents,
                        shuffle_languages=shuffle_languages,
                        src_lang = src_lang,
                        dev_size = dev_size,
                        shuffled_size = shuffled_size,
                        unshuffled_size = unshuffled_size,
                        drop_duplicates = drop_duplicates,
                        ).shuffle(seed=42)

    dataset = dataset.map(lambda sample: qa_tokenize(sample, tokenizer),
                        batched=True,
                        batch_size=8,
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
    shuffle_ents = True
    shuffle_languages = ['it']
    src_lang = 'en'
    tokenizer_name = 'bert-base-multilingual-cased'
    tokenizer_name = 'microsoft/mdeberta-v3-base'
    drop_duplicates = True
    dataset = build_tasteset(json_path,
                             shuffle_ents = shuffle_ents,
                             shuffle_languages = shuffle_languages,
                             src_lang = src_lang,
                             tokenizer_name = tokenizer_name,
                             drop_duplicates = drop_duplicates,
                             )
    lang_id = '-'.join(shuffle_languages)
    suffix = 'drop_duplicates' if drop_duplicates == True else ''
    output_path = os.path.join(dir_path, f'.{src_lang}/{tokenizer_name.split("/")[-1]}_{lang_id}_{suffix}')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # print(dataset)
    # print(dataset['train'])
    # print(dataset['train'][0])
    
    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    main()