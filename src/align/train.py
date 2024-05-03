import warnings
import torch
torch.set_printoptions(linewidth=100000, threshold=100000)
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, BertModel, PretrainedConfig, BertPreTrainedModel
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import data_loader, SquadEvaluator, TASTEset, save_local_model
from aligner_pt import BertAligner
import re
from datetime import datetime
import os
import argparse
import json
# import sys
# sys.path.append('/home/pgajo/food/bert_crf')
# from crf_models.bert_crf import BertCrf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--data_path')
    parser.add_argument('--use_bert_aligner', action='store_true', default=False,
                    help='Use BERT aligner if this flag is set (default: %(default)s)')
    args = parser.parse_args()

    model_name = args.model_path
    data_name = args.data_path

    langs = re.search(r'([a-z]{2}-[a-z]{2})', data_name).group(1)

    data_name_simple = data_name.split('/')[-1]

    data = TASTEset.from_disk(data_name)

    batch_size = 16
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=320,
                        )
    device = 'cuda'
    bertaligner = args.use_bert_aligner
    if bertaligner:
        model = BertAligner.from_pretrained(model_name,
                                        output_hidden_states=True).to(device)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    
    lr = 3e-5
    eps = 1e-8
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                lr = lr,
                                eps = eps
                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )
    bottom_out_non_context = 0
    epochs = 3
    for epoch in range(epochs):
        # train
        epoch_train_loss = 0
        model.train()
        split = 'train'
        progbar = tqdm(enumerate(dataset[split]),
                                total=len(dataset[split]),
                                desc=f"{split} - epoch {epoch + 1}")
        print('model_name:', model_name)
        print('data_name:', data_name)
        columns = [
                    'input_ids',
                    'token_type_ids',
                    'attention_mask',
                    'start_positions',
                    'end_positions'
                    ]
        for i, batch in progbar:
            input = {k: batch[k].to(device) for k in columns}
            outputs = model(**input) # ['loss', 'start_logits', 'end_logits']
            if bottom_out_non_context:
                for i in range(len(outputs['start_logits'])):
                    outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], -10000)
                    outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], -10000)
            
            loss = outputs['loss']#.mean()
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
        print('model_name:', model_name)
        print('data_name:', data_name)
        for i, batch in progbar:
            input = {k: batch[k].to(device) for k in columns}
            with torch.inference_mode():
                outputs = model(**input)
            loss = outputs['loss']#.mean()
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
    
    metrics = evaluator.print_metrics()

    if bertaligner:
        model_name_simple = f"{model_name.split('/')[-1]}_BertAligner"
    else:
        model_name_simple = model_name.split('/')[-1]
    
    data_name = re.sub('.json', '', data_name.split('/')[-1]) # remove extension if local path

    results_path = f'/home/pgajo/food/results/alignment/{langs}/dev/{data_name}/{model_name_simple}/'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    model_dir = f'/home/pgajo/food/models/alignment/{langs}/{data_name}/{model_name_simple}/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_name = f"E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}_ME{epochs}_{date_time}"
    csv_save_path = os.path.join(results_path, save_name)
    print('Saving metrics to:', csv_save_path)
    evaluator.save_metrics_to_csv(csv_save_path)

    model_save_dir = os.path.join(model_dir, f"{model_name_simple}_{data_name_simple}_ME{epochs}_{date_time}")
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    evaluator.save_metrics_to_csv(os.path.join(model_save_dir, 'metrics'))
    save_local_model(model_save_dir, model, tokenizer)

    model_info = {
        'model_name': model_name,
        'batch_size': batch_size,
        'learning_rate': 3e-5,
        'epsilon': 1e-8,
        'data_name': data_name,
        'len_train': len(dataset['train']),
        'len_dev': len(dataset['dev']),
        'max_epochs': epochs,
        'date_time': date_time,
        'tokenizer_type': str(type(tokenizer).__name__),
        'bottom_out_non_context': bottom_out_non_context,
        'use_bert_aligner': args.use_bert_aligner,
    }

    info_path = os.path.join(model_save_dir, 'model_info.json')
    with open(info_path, 'w', encoding='utf8') as f:
        json.dump(model_info, f, ensure_ascii = False)
    
    # this needs to be the last print from this script
    # for the bash script to pick up the model for test evaluation
    print(model_save_dir)
    

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
