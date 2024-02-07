import os
from utils import TASTEset
import warnings
import argparse
from transformers import AutoTokenizer

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
    args.unshuffled_size = 1
    args.shuffled_size = 0
    args.drop_duplicates = True

    args.input = '/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL_localized_uom.json'

    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TASTEset.from_json(args.input,
        tokenizer =  tokenizer,
        shuffle_languages = ['it'],
        src_lang = 'en',
        dev_size = 0.2,
        shuffled_size = args.shuffled_size,
        unshuffled_size = args.unshuffled_size,
        # drop_duplicates = True,
        debug_dump = True,
        # n_rows = 10,
        aligned = False,
        )
    
    # print(dataset)
    # print(dataset['train'][0:5])
    # print(dataset.__class__)
    
    # lang_id = '-'.join(args.shuffle_languages.split())
    # suffix = 'drop_duplicates' if args.drop_duplicates == True else 'keep_duplicates'

    # if not args.output:
    #     output_path = os.path.join(os.path.dirname(args.input),
    #                 f'.{args.src_lang}/{args.tokenizer_name.split("/")[-1]}_{lang_id}_{suffix}_shuffled_size_{args.shuffled_size}')
    #     if not os.path.isdir(output_path):
    #         os.makedirs(output_path)
    
    # dataset.save_to_disk(output_path)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()