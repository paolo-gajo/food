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
    tokenizer_name = 'bert-base-multilingual-cased'
    model_name = 'bert-base-multilingual-cased'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'
    # model_name = 'microsoft/mdeberta-v3-base'
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    push_model_description = ''

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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
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
    results_path = f'/home/pgajo/working/food/results/xl-wa/{lang_id}'
    data = XLWADataset(
        data_path,
        tokenizer,
        languages = languages,
        # n_rows=20,
        )
    
    data_name = data.name
    # data = DatasetDict.load_from_disk(data_path) # load prepared tokenized dataset
    # print('max_num_tokens', [len(data['train'][i]['input_ids']) for i in range(0)])
    batch_size = 32
    dataset = data_loader(data,
                        batch_size,
                          n_rows = 500,
                        )

    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = 'cuda'
    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=2e-5,
                                eps=1e-8
                                )

    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )

    epochs = 1

    for epoch in range(epochs):
        # train
        epoch_train_loss = 0
        model.train()
        split = 'train'
        progbar = tqdm(enumerate(dataset[split]),
                                total=len(dataset[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        for i, batch in progbar:
            outputs = model(**batch) # ['loss', 'start_logits', 'end_logits']
            loss = outputs[0].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            loss_tmp = round(epoch_train_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split)

        evaluator.evaluate(split, epoch)
        epoch_train_loss /= len(dataset[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_train_loss

        evaluator.print_metrics(current_epoch = epoch, current_split = split)

        # eval on dev
        epoch_dev_loss = 0
        model.eval()
        split = 'dev'
        progbar = tqdm(enumerate(dataset[split]),
                                total=len(dataset[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        for i, batch in progbar:
            with torch.inference_mode():
                outputs = model(**batch)
            loss = outputs[0].mean()
            epoch_dev_loss += loss.item()
            loss_tmp = round(epoch_dev_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split)
        
        evaluator.evaluate(split, epoch)
        epoch_dev_loss /= len(dataset[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_dev_loss

        evaluator.print_metrics(current_epoch = epoch, current_split = split)

        evaluator.store_metrics()

        if evaluator.stop_training:
            print(f'Early stopping triggered on epoch {epoch}. \
                \nBest epoch: {evaluator.epoch_best}.')                                               
            break

    evaluator.print_metrics()
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    evaluator.save_metrics_to_csv(results_path)

    # model save folder
    model_dir = '/home/pgajo/working/food/src/word_alignment/models'
    model_save_dir = os.path.join(model_dir, f"{data_name}/{model_name.split('/')[-1]}_{evaluator.f1_dev_best}")
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    save_local_model(model_save_dir, model, tokenizer)
    
    suffix = f"{data_name}_{evaluator.epoch_best}_epochs_{round(evaluator.f1_dev_best, ndigits=2)}"
    push_model_repo_to_hf(model_save_dir, model_name=model_name, suffix=suffix)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()