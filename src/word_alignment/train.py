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

    # model_repo = 'bert-base-multilingual-cased' 
    model_repo = 'pgajo/mbert-xlwa-en-it' # mbert fine-tuned on xlwa-en-it

    # xl-wa
    # data_repo = 'pgajo/xlwa_en-it_mbert' 
    
    # tasteset
    # data_repo = 'pgajo/EW-TT-PE_U1_S0_DROP1_mbert' # unshuffled
    data_repo = 'pgajo/EW-TT-PE_U0_S1_DROP1_mbert' # shuffled

    # data_repo = 'pgajo/EW-TT-MT_LOC_U1_S0_DROP1_mbert' # unshuffled
    # data_repo = 'pgajo/EW-TT-MT_LOC_U0_S1_DROP1_mbert' # shuffled

    # model_repo = 'microsoft/mdeberta-v3-base'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it' # mdeberta fine-tuned on xlwa-en-it
    
    # xl-wa
    # data_repo = 'pgajo/xlwa_en-it_mdeberta'

    # tasteset
    # data_repo = 'pgajo/EW-TT-PE_U1_S0_DROP1_mdeberta' # unshuffled
    # data_repo = 'pgajo/EW-TT-PE_U0_S1_DROP1_mdeberta' # shuffled

    # data_repo = 'pgajo/EW-TT-MT_LOC_U1_S0_DROP1_mdeberta' # unshuffled
    # data_repo = 'pgajo/EW-TT-MT_LOC_U0_S1_DROP1_mdeberta' # shuffled
    
    data = TASTEset.from_datasetdict(data_repo)

    # args.input = '/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL_localized_uom.json'
    # data = TASTEset.from_json(
    #     args.input,
    #     model_repo,
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
    model = AutoModelForQuestionAnswering.from_pretrained(model_repo)
    device = 'cuda'
    model = torch.nn.DataParallel(model).to(device)
    
    lr = 3e-5
    eps = 1e-8
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                lr = lr,
                                eps = eps
                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
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
        print('model_repo:', model_repo)
        print('data_repo:', data_repo)
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
        print('model_repo:', model_repo)
        print('data_repo:', data_repo)
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
    
    results_path = '/home/pgajo/working/food/results'
    # model save folder
    model_dir = './models'
    model_name = model_repo.split('/')[-1].split('_')[0]
    data_name = re.sub('\..*', '', data_repo.split('/')[-1]) # remove extension if local path
    model_results_path = os.path.join(results_path, data_name)
    save_name = f"{model_name}_{data_name}_E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}"
    save_name = save_name.replace('bert-base-multilingual-cased', 'mbert')
    save_name = save_name.replace('mdeberta-v3-base', 'mdeberta')

    evaluator.save_metrics_to_csv(os.path.join(model_results_path, save_name))

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
    Model: {model_repo}\n
    Dataset: {data.name}\n
    Unshuffled ratio: {re.findall(u_ratio, data_repo)}\n
    Shuffled ratio: {re.findall(s_ratio, data_repo)}\n
    Best exact match epoch: {evaluator.epoch_best}\n
    Best exact match: {str(round(evaluator.exact_dev_best, ndigits=2))}\n
    Best epoch: {evaluator.epoch_best}\n
    Drop duplicates: {re.findall(drop_flag, data_repo)}\n
    Max epochs = {epochs}\n
    Optimizer lr = {lr}\n
    Optimizer eps = {eps}\n
    Batch size = {batch_size}\n
    Dataset path = {data_repo}\n
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
