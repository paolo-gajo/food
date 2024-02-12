from utils import TASTEset, push_dataset_card
import warnings
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='dummy argument to make the script work on Google Colab')
    parser.add_argument('-i', '--input', default='/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TASTE_en-it_DEEPL.json', help='path of the input json dataset')
    parser.add_argument('-o', '--output', default='', help='path of the input json dataset')
    parser.add_argument('-l', '--shuffle_languages', default='it', help='space-separated 2-character codes of the dataset target languages to shuffle')
    parser.add_argument('-src', '--src_lang', default='en', help='space-separated 2-character code of the dataset source language')
    parser.add_argument('-t', '--tokenizer_name', default='bert-base-multilingual-cased', help='tokenizer to use')
    parser.add_argument('-d', '--drop_duplicates', default=True, help='if True (default=True), drop rows with the same answer')
    parser.add_argument('-ss', '--shuffled_size', default=1, help='length multiplier for the number of shuffled instances (default=1)')
    parser.add_argument('-us', '--unshuffled_size', default=1, help='length multiplier for the number of unshuffled instances (default=1)')
    parser.add_argument('-ds', '--dev_size', default=1, help='size of the dev split (default=0.2)')

    args = parser.parse_args()
    # args.input = '/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL.json'
    args.unshuffled_size = 0
    args.shuffled_size = 1
    args.drop_duplicates = True

    # tokenizer_name = 'bert-base-multilingual-cased'
    # tokenizer_name = 'bert-large-uncased'
    tokenizer_name = 'microsoft/mdeberta-v3-base'

    # args.input = '/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TT-PE.json'
    args.input = '/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_LOC.json'
    dataset = TASTEset.from_json(
        args.input,
        tokenizer_name,
        tgt_langs = ['it'],
        src_lang = 'en',
        dev_size = 0.2,
        shuffled_size = args.shuffled_size,
        unshuffled_size = args.unshuffled_size,
        shuffle_type = 'ingredient',
        )
    
    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
        'bert-large-uncased': 'bert-large-uncased'
    }

    save_name = f"{args.input.split('/')[-1].replace('.json', '')}_U{dataset.unshuffled_size}_S{dataset.shuffled_size}_DROP{str(int(dataset.drop_duplicates))}_{tokenizer_dict[tokenizer_name]}"
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