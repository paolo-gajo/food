import os
from utils import XLWADataset
from transformers import AutoTokenizer

def main():
    data_path = '/home/pgajo/working/food/data/XL-WA/data'
    languages = [
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
    tokenizer_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = XLWADataset(
        data_path,
        tokenizer,
        languages = languages,
        # n_rows=20,
        )
    print(dataset)

    # lang_id = '-'.join(lang_list)
    # output_path = os.path.join(data_path, f'.formatted/{lang_id}')
    # if not os.path.isdir(output_path):
    #     os.makedirs(output_path)
    
    # dataset.save_to_disk(output_path)

if __name__ == '__main__':
    main()