import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import data_loader, SquadEvaluator, TASTEset
import pandas as pd
import re
from aligner_pt import AlignerLoss, BertAligner
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', default='/home/pgajo/food/models/alignment/en-it/best/mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-04-02-17-37-26_TEST53.0')
    parser.add_argument('-d', '--data_test_path', default='/home/pgajo/food/datasets/alignment/en-it/BertTokenizerFast/GZ-GOLD_301_BertTokenizerFast_en-it')
    parser.add_argument('-t', '--types', default=False, action='store_true', help='Add a "_test" suffix to the repo name')
    args = parser.parse_args()

    langs = re.search(r'([a-z]{2}-[a-z]{2})(-[a-z]{2})?', args.model_path).group(0)
    print(langs)

    model_name = args.model_path
    data_repo_test = args.data_test_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_test = TASTEset.from_disk(data_repo_test)

    batch_size = 128
    dataset_test = data_loader(data_test, 
                        batch_size,
                        # n_rows=100
                        )

    device = 'cuda'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    # model = BertAligner.from_pretrained(model_name, output_hidden_states=True).to(device)
    
    # model = torch.nn.DataParallel(model).to(device)

    evaluator = SquadEvaluator(tokenizer,
                            model,
                            load("squad_v2"),
                            )

    epoch = 0
    # eval on test
    epoch_test_loss = 0
    model.eval()
    split = 'test'
    progbar = tqdm(enumerate(dataset_test[split]),
                            total=len(dataset_test[split]),
                            desc=f"{split} - epoch {epoch + 1}")
    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
    for i, batch in progbar:
        with torch.inference_mode():
            input = {k: batch[k].to('cuda') for k in columns}
            outputs = model(**input)
            # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
            # for i in range(len(outputs['start_logits'])):
            #     outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], input['token_type_ids'][i]-10000)
            #     outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], input['token_type_ids'][i]-10000)
        # loss = outputs[0].mean()
        loss = outputs['loss'].mean()
        epoch_test_loss += loss.item()
        loss_tmp = round(epoch_test_loss / (i + 1), 4)
        progbar.set_postfix({'Loss': loss_tmp})
        
        evaluator.get_eval_batch(outputs, batch, split, type_labels = True)

    results_path = f'/home/pgajo/food/results/alignment/{langs}/test'
    # model save folder
    model_name_simple = model_name.split('/')[-1]
    data_name = data_repo_test.split('/')[-1]
    data_results_path = os.path.join(results_path, data_name)
    if not os.path.isdir(data_results_path):
        os.makedirs(data_results_path)
    save_name = f"{model_name_simple}_E{evaluator.epoch_best}_{split.upper()}{str(round(evaluator.exact_dev_best, ndigits=0))}"

    metrics_save_path = os.path.join(results_path, data_name, save_name)
    if not os.path.exists(metrics_save_path):
        os.makedirs(metrics_save_path)
    
    evaluator.evaluate(model, split, epoch, eval_metric='test', model_name = model_name_simple)
    epoch_test_loss /= len(dataset_test[split])
    evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

    metrics = evaluator.print_metrics(current_epoch = epoch, current_split = split).to_dict()
    evaluator.store_metrics()

    metrics[0]['model_path'] = str(args.model_path)
    metrics[0]['data_test_path'] = str(args.data_test_path)
    

    with open(os.path.join(metrics_save_path, 'metrics.json'), 'w', encoding='utf8') as f:
        json.dump(metrics[0], f, ensure_ascii = False)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()