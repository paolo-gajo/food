import sys
sys.path.append('/home/pgajo/food/src')
from utils import TASTEset, push_dataset_card
import warnings
import os

def main():
    unshuffled_size = 1
    shuffled_size = 0

    tokenizer_name = 'bert-base-multilingual-cased'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'

    data_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced.json'
    
    dataset = TASTEset.from_json(
        data_path,
        tokenizer_name,
        src_lang = 'en',
        tgt_langs = ['it'], # N.B.: the actual source language in GZ is italian, but our models were trained on english to predict an italian target
        dev_size = 0,
        shuffled_size = shuffled_size,
        unshuffled_size = unshuffled_size,
        # aligned = False,
        # debug_dump = True,
        drop_duplicates = False,
        label_studio = True,
        inverse_languages = True,
        # verbose = True,
        n_rows = 106
        
        )
    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }
    save_name = f"{data_path.split('/')[-1].replace('.json', '')}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_T{dataset.shuffle_type}_P{dataset.shuffle_probability}_DROP{str(int(dataset.drop_duplicates))}_{tokenizer_dict[tokenizer_name]}_{dataset.src_lang}-{''.join(dataset.tgt_langs)}_INV{int(dataset.inverse_languages)}_align"
    repo_id = f"pgajo/{save_name}"
    print('repo_id:', repo_id)
    # dataset.push_to_hub(repo_id)
    # dataset_summary = f'''
    # Tokenizer: {tokenizer_dict[tokenizer_name]}\n
    # Dataset: {dataset.name}\n
    # Unshuffled ratio: {dataset.unshuffled_size}\n
    # Shuffled ratio: {dataset.shuffled_size}\n
    # Shuffle probability: {dataset.shuffle_probability}\n
    # Drop duplicates: {dataset.drop_duplicates}\n
    # Dataset path = {dataset.data_path}\n
    # '''
    # push_dataset_card(repo_id, dataset_summary=dataset_summary)
    datasets_dir_path = '/home/pgajo/food/datasets/alignment'
    dataset.save_to_disk(os.path.join(datasets_dir_path, save_name))


    # dataset.name = data_path.split('/')[-1].replace('.json', '')
    # save_name = f"{tokenizer_dict[tokenizer_name]}_{dataset.name}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_DROP{str(int(dataset.drop_duplicates))}"
    # repo_id = f"pgajo/{save_name}_types"
    # print('repo_id:', repo_id)
    # local_dir = data_path.replace('.json', '')
    # if not os.path.isdir(local_dir):
    #     os.makedirs(local_dir)
    # # dataset.save_to_disk(local_dir)
    # dataset.push_to_hub(repo_id)
    # dataset_summary = f'''
    # Tokenizer: {tokenizer_dict[tokenizer_name]}\n
    # Dataset: {dataset.name}\n
    # Unshuffled ratio: {dataset.unshuffled_size}\n
    # Shuffled ratio: {dataset.shuffled_size}\n
    # Drop duplicates: {dataset.drop_duplicates}\n
    # Dataset path = {dataset.data_path}\n
    # '''
    # push_dataset_card(repo_id, dataset_summary=dataset_summary, model_metrics = '')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()