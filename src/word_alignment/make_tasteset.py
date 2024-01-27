import os
from utils import prep_tasteset, qa_tokenize
from transformers import AutoTokenizer
import warnings
import argparse

def build_tasteset(data_path,
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
                        shuffle_languages=shuffle_languages,
                        src_lang = src_lang,
                        dev_size = dev_size,
                        shuffled_size = shuffled_size,
                        unshuffled_size = unshuffled_size,
                        drop_duplicates = drop_duplicates,
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
    parser = argparse.ArgumentParser()
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

    args.unshuffled_size = 1
    args.shuffled_size = 0
    args.drop_duplicates = True

    dataset = build_tasteset(args.input,
                             shuffle_languages = args.shuffle_languages,
                             src_lang = args.src_lang,
                             tokenizer_name = args.tokenizer_name,
                             drop_duplicates = args.drop_duplicates,
                             shuffled_size = args.shuffled_size,
                             unshuffled_size = args.unshuffled_size,
                             dev_size = args.dev_size
                             )
    
    lang_id = '-'.join(args.shuffle_languages.split())
    suffix = 'drop_duplicates' if args.drop_duplicates == True else 'keep_duplicates'

    if not args.output:
        output_path = os.path.join(os.path.dirname(args.input),
                    f'.{args.src_lang}/{args.tokenizer_name.split("/")[-1]}_{lang_id}_{suffix}_shuffled_size_{args.shuffled_size}')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    
    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()