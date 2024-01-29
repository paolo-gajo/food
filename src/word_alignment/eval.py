import warnings
import os
import torch
from tqdm.auto import tqdm
from datasets import DatasetDict
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset, XLWADataset, push_model_repo_to_hf, save_local_model
from datetime import datetime

def main():
    model_name = '/home/pgajo/working/food/src/word_alignment/models/xlwa/bert-base-multilingual-cased_0'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'
    # model_name = 'microsoft/mdeberta-v3-base'
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

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

    lang_id = '-'.join(languages)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # data_path = f'/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/.en/{tokenizer_name.split("/")[-1]}_it_drop_duplicates'
    # results_path = f'/home/pgajo/working/food/results/tasteset/{lang_id}'
    # data = TASTEset.from_json(
    #         args.input,
    #         shuffle_languages=['it'],
    #         src_lang = 'en',
    #         tokenizer_name = 'bert-base-multilingual-cased',
    #         dev_size=0.2,
    #         shuffled_size = 1,
    #         unshuffled_size = 1,
    #         drop_duplicates = True,
    #         n_rows=10,
    #         )

    data_path = f'//home/pgajo/working/food/data/XL-WA/data'
    data = XLWADataset(
        data_path,
        tokenizer,
        languages = languages,
        splits=['test']
        )
    
    data_name = data.name
    # data = DatasetDict.load_from_disk(data_path) # load prepared tokenized dataset
    # print('max_num_tokens', [len(data['train'][i]['input_ids']) for i in range(0)])
    batch_size = 32
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=100
                        )

    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = 'cuda'
    model = torch.nn.DataParallel(model).to(device)

    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )

    epoch = 0
    # eval on test
    epoch_test_loss = 0
    model.eval()
    split = 'test'
    progbar = tqdm(enumerate(dataset[split]),
                            total=len(dataset[split]),
                            desc=f"{split} - epoch {epoch + 1}")
    for i, batch in progbar:
        with torch.inference_mode():
            outputs = model(**batch)
        loss = outputs[0].mean()
        epoch_test_loss += loss.item()
        loss_tmp = round(epoch_test_loss / (i + 1), 4)
        progbar.set_postfix({'Loss': loss_tmp})
        
        evaluator.get_eval_batch(outputs, batch, split)
    
    evaluator.evaluate(split, epoch)
    epoch_test_loss /= len(dataset[split])
    evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

    evaluator.print_metrics(current_epoch = epoch, current_split = split)
    evaluator.store_metrics()
    evaluator.append_test_metrics(model_name)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()