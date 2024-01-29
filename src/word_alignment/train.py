import warnings
import os
import torch
from tqdm.auto import tqdm
from datasets import DatasetDict
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset, XLWADataset, push_model_repo_to_hf, save_local_model, push_card
from datetime import datetime
from huggingface_hub import HfApi

def main():
    model_name = 'bert-base-multilingual-cased'
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
    
    data_path = f'/home/pgajo/working/food/data/EW-TASTE_en-it_DEEPL.json'
    results_path = f'/home/pgajo/working/food/results/tasteset/{lang_id}'
    data = TASTEset.from_json(
            data_path,
            tokenizer_name = model_name,
            shuffle_languages=['it'],
            src_lang = 'en',
            dev_size = 0.2,
            shuffled_size = 0,
            unshuffled_size = 1,
            # drop_duplicates = False,
            debug_dump = True,
            # n_rows=200,
            )

    # data_path = f'//home/pgajo/working/food/data/XL-WA/data'
    # results_path = f'/home/pgajo/working/food/results/xl-wa/{lang_id}'
    # data = XLWADataset(
    #     data_path,
    #     tokenizer,
    #     languages = languages,
    #     # n_rows=20,
    #     )
    
    # data = DatasetDict.load_from_disk(data_path) # load prepared tokenized dataset
    # print('max_num_tokens', [len(data['train'][i]['input_ids']) for i in range(0)])
    batch_size = 32
    dataset = data_loader(data,
                        batch_size,
                        )

    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = 'cuda'
    model = torch.nn.DataParallel(model).to(device)
    
    lr = 3e-5
    eps=1e-8
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=lr,
                                eps=eps
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    evaluator.save_metrics_to_csv(results_path)

    model_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }

    if not hasattr(data, 'unshuffled_size'):
        data.unshuffled_size = 1
    if not hasattr(data, 'shuffled_size'):
        data.shuffled_size = 0
        
    # model save folder
    model_dir = '/home/pgajo/working/food/src/word_alignment/models'
    save_name = f"{model_dict[model_name]}_{data.name}_U{data.unshuffled_size}_S{data.shuffled_size}_E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}_DROP{str(int(data.drop_duplicates))}"
    model_save_dir = os.path.join(model_dir, f"{data.name}/{save_name}")
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    evaluator.save_metrics_to_csv(os.path.join(model_save_dir, 'metrics'))
    save_local_model(model_save_dir, model, tokenizer)
    
    model_description = f'''
    Model: {model_dict[model_name]}
    Dataset: {data.name}
    Unshuffled ratio: {data.unshuffled_size}
    Shuffled ratio: {data.shuffled_size}
    Best exact match epoch: {evaluator.epoch_best}
    Best exact match: {str(round(evaluator.exact_dev_best, ndigits=2))}
    Drop duplicates: {data.drop_duplicates}
    Optimizer lr = {lr}
    Optimizer eps = {eps}
    Batch size = {batch_size}
    '''
    open(f'{model_save_dir}/model_description.txt', 'w', encoding='utf8').write(model_description)

    repo_id = f"pgajo/{save_name}"

    api = HfApi()
    token = os.environ['HF_WRITE_TOKEN']
    api.create_repo(repo_id, token=token)
    push_card(repo_id=repo_id,
            model_name=model_name,
            model_description=model_description,
            )
    api.upload_folder(repo_id=repo_id, folder_path=model_save_dir, token=token)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()