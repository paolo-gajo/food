from utils import TASTEset, push_dataset_card
import warnings
import argparse
import json

def main():
    unshuffled_size = 1
    shuffled_size = 0
    drop_duplicates = True

    tokenizer_name = 'bert-base-multilingual-cased'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'

    data_path = '/home/pgajo/working/food/data/GZ-GOLD-NER_for_sample_generation.json'
    
    dataset = TASTEset.from_json(
        data_path,
        tokenizer_name,
        shuffle_languages = ['en'],
        src_lang = 'it',
        dev_size = 0.2,
        shuffled_size = shuffled_size,
        unshuffled_size = unshuffled_size,
        aligned = False,
        debug_dump = True,
        )
    
    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }
    
    dataset.name = 'GZ-ALIGN'
    save_name = f"{tokenizer_dict[tokenizer_name]}_{dataset.name}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_DROP{str(int(dataset.drop_duplicates))}"
    repo_id = f"pgajo/{save_name}"
    print('repo_id:', repo_id)
    dataset.push_to_hub(repo_id)
    dataset_summary = f'''
    Tokenizer: {tokenizer_dict[tokenizer_name]}\n
    Dataset: {dataset.name}\n
    Unshuffled ratio: {dataset.unshuffled_size}\n
    Shuffled ratio: {dataset.shuffled_size}\n
    Drop duplicates: {dataset.drop_duplicates}\n
    Dataset path = {dataset.data_path}\n
    '''
    push_dataset_card(repo_id, dataset_summary=dataset_summary)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()