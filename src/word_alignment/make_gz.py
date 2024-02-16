from utils import TASTEset, push_dataset_card
import warnings
import os

def main():
    unshuffled_size = 1
    shuffled_size = 0

    # tokenizer_name = 'bert-base-multilingual-cased'
    tokenizer_name = 'microsoft/mdeberta-v3-base'

    data_path = '/home/pgajo/working/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105.json'
    
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
        )
    # return
    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }

    dataset.name = data_path.split('/')[-1].replace('.json', '')
    save_name = f"{tokenizer_dict[tokenizer_name]}_{dataset.name}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_DROP{str(int(dataset.drop_duplicates))}"
    repo_id = f"pgajo/{save_name}_types"
    print('repo_id:', repo_id)
    local_dir = data_path.replace('.json', '')
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    # dataset.save_to_disk(local_dir)
    dataset.push_to_hub(repo_id)
    dataset_summary = f'''
    Tokenizer: {tokenizer_dict[tokenizer_name]}\n
    Dataset: {dataset.name}\n
    Unshuffled ratio: {dataset.unshuffled_size}\n
    Shuffled ratio: {dataset.shuffled_size}\n
    Drop duplicates: {dataset.drop_duplicates}\n
    Dataset path = {dataset.data_path}\n
    '''
    push_dataset_card(repo_id, dataset_summary=dataset_summary, model_metrics = '')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()