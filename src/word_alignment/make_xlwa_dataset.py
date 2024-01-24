import os
from utils import prep_xl_wa, qa_tokenize
from transformers import AutoTokenizer

def build_xl_wa(data_path, lang_list):
    lang_id = '-'.join(lang_list)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    dataset = prep_xl_wa(data_path,
                        lang_list,
                        tokenizer,
                        #   n_rows = 20,
                        src_context = True,
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

    data_train_path = os.path.join(data_path, f'.ready/{lang_id}')
    if not os.path.isdir(data_train_path):
        os.mkdir(data_train_path)
        
    dataset.save_to_disk(data_train_path)

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

if __name__ == '__main__':
    main()