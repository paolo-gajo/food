import os
from utils import prep_xl_wa, qa_tokenize
from transformers import AutoTokenizer

def build_xl_wa(data_path,
                lang_list,
                n_rows = None
                ):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    dataset = prep_xl_wa(data_path,
                        lang_list,
                        tokenizer,
                        n_rows = n_rows,
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
    data_path = '/home/pgajo/working/food/data/XL-WA/data'
    
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
    
    dataset = build_xl_wa(data_path,
    lang_list,
    # n_rows=20,
    )
    src_lang = 'en'
    lang_id = '-'.join(lang_list)
    output_path = os.path.join(data_path, f'.{src_lang}/{lang_id}')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # print(dataset)
    # print(dataset['train'])
    # print(dataset['train'][0])
    
    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    main()