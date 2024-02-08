import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset, XLWADataset, save_local_model, push_model_card
from huggingface_hub import HfApi
import pandas as pd
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default=False, action='store_true', help='Add a "_test" suffix to the repo name')
    parser.add_argument('-f', '--file', default='/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL_localized_uom.json')
    args = parser.parse_args()

    model_name = 'bert-base-multilingual-cased'
    # model_name = 'microsoft/mdeberta-v3-base'
    
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

    args.file = 'pgajo/mbert_TASTEset_U0_S1_DROP1'
    # args.file = 'pgajo/xlwa_en-it'
    data = TASTEset.from_datasetdict(args.file)

    # args.input = '/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL_localized_uom.json'
    # data = TASTEset.from_json(
    #     args.input,
    #     model_name,
    #     shuffle_languages = ['it'],
    #     src_lang = 'en',
    #     dev_size = 0.2,
    #     shuffled_size = 0,
    #     unshuffled_size = 1,
    #     )

    batch_size = 32
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=320,
                        )
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = 'cuda'
    model = torch.nn.DataParallel(model).to(device)
    
    lr = 3e-5
    eps = 1e-8
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                lr = lr,
                                eps = eps
                                )

    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )

    epochs = 10
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            loss_tmp = round(epoch_train_loss / (i + 1), 4)
            progbar.set_postfix({'Loss': loss_tmp})
            
            evaluator.get_eval_batch(outputs, batch, split)

        evaluator.evaluate(model, split, epoch)
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
        
        evaluator.evaluate(model, split, epoch)
        epoch_dev_loss /= len(dataset[split])
        evaluator.epoch_metrics[f'{split}_loss'] = epoch_dev_loss

        evaluator.print_metrics(current_epoch = epoch, current_split = split)

        evaluator.store_metrics()

        if evaluator.stop_training:
            print(f'Early stopping triggered on epoch {epoch}. \
                \nBest epoch: {evaluator.epoch_best}.')                                               
            break
    
    evaluator.print_metrics()
    
    results_path = f'/home/pgajo/working/food/results/tasteset/{lang_id}'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    evaluator.save_metrics_to_csv(results_path)

    model_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }
        
    # model save folder
    model_dir = './models'
    filename_simple = re.sub('\..*', '', args.file.split('/')[-1]) # remove extension if local path
    save_name = f"{filename_simple}_E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}"
    # filename_simple = f"{data.name}_U{data.unshuffled_size}_S{data.shuffled_size}_DROP{data.drop_duplicates}"
    # save_name = f"{filename_simple}_E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}"
    if args.test:
        save_name = save_name + "_test" # comment if not testing
    model_save_dir = os.path.join(model_dir, f"{data.name}/{save_name}")
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    evaluator.save_metrics_to_csv(os.path.join(model_save_dir, 'metrics'))
    save_local_model(model_save_dir, model, tokenizer)

    repo_id = f"pgajo/{save_name}"
    print('repo_id', repo_id)
    api = HfApi(token = os.environ['HF_TOKEN'])
    api.create_repo(repo_id)
    df_desc = pd.DataFrame(evaluator.metrics).round(2)
    df_desc.index += 1
    df_desc.index.name = 'epoch'
    df_desc = df_desc.to_markdown()
    u_ratio = '(?<=_U)\d'
    s_ratio = '(?<=_S)\d'
    drop_flag = '(?<=_DROP)\d'
    model_description = f'''
    Model: {model_dict[model_name]}\n
    Dataset: {data.name}\n
    Unshuffled ratio: {re.search(u_ratio, 'pgajo/mbert_TASTEset_U0_S1_DROP1').group()}\n
    Shuffled ratio: {re.search(s_ratio, 'pgajo/mbert_TASTEset_U0_S1_DROP1').group()}\n
    Best exact match epoch: {evaluator.epoch_best}\n
    Best exact match: {str(round(evaluator.exact_dev_best, ndigits=2))}\n
    Drop duplicates: {re.search(drop_flag, 'pgajo/mbert_TASTEset_U0_S1_DROP1').group()}\n
    Epochs = {epochs}\n
    Optimizer lr = {lr}\n
    Optimizer eps = {eps}\n
    Batch size = {batch_size}\n
    Dataset path = {args.file}\n
    '''
    push_model_card(repo_id=repo_id,
            model_description=model_description,
            results=df_desc,
            template_path='/home/pgajo/modelcardtemplate.md'
            )
    api.upload_folder(repo_id=repo_id, folder_path=model_save_dir) 

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
