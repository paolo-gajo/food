import os
from utils import XLWADataset, push_dataset_card
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
    # tokenizer_name = 'bert-base-multilingual-cased'
    tokenizer_name = 'microsoft/mdeberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = XLWADataset(
        data_path,
        tokenizer,
        languages = languages,
        # n_rows=20,
        )

    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }
    save_name = f"{tokenizer_dict[tokenizer_name]}_{dataset.name}"
    repo_id = f"pgajo/{save_name}"
    print('repo_id:', repo_id)
    dataset.push_to_hub(repo_id)
    dataset_summary = f'''
    Tokenizer: {tokenizer_dict[tokenizer_name]}\n
    Dataset: {dataset.name}\n
    Dataset path = {data_path}\n
    '''
    push_dataset_card(repo_id, dataset_summary=dataset_summary)

if __name__ == '__main__':
    main()