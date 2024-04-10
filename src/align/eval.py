import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils import data_loader, SquadEvaluator, TASTEset
import pandas as pd
import re
from aligner_pt import AlignerLoss, BertAligner
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--data_test_path', default='/home/pgajo/food/datasets/alignment/it/mbert/GZ-GOLD-NER-ALIGN_105_spaced_U1_S0_Trecipe_P0_DROP0_en-it_INV1')
    args = parser.parse_args()

    # mbert
    # P = 0
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME10_2024-03-29-12-29-26'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-13-43-01'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-14-10-55'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-15-08-20'

    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME10_2024-03-29-13-25-43'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-13-52-16'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-14-18-14'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-15-14-47'

    # P = 0.2
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-16-08-15'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-16-14-33'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-16-20-50'

    # P = 0.3
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.3_DROP0_ME3_2024-03-29-16-27-19'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.3_DROP0_ME3_2024-03-29-16-33-38'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.3_DROP0_ME3_2024-03-29-16-41-08'

    # P = 0.4
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.4_DROP0_ME3_2024-03-29-16-48-55'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.4_DROP0_ME3_2024-03-29-16-55-13'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.4_DROP0_ME3_2024-03-29-17-01-35'

    # mbert-xl-wa-en-it
    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/mbert_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0'

    # mdeberta
    # P = 0
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-20-55-50'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-21-04-40'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-21-14-54'

    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-21-24-08'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-21-32-31'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-21-43-23'

    # P = 0.2
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-22-01-50'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-22-10-14'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base/mdeberta-v3-base_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.2_DROP0_ME3_2024-03-29-22-18-36'

    # mdeberta xl-wa
    # P = 0
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-20-00-14'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-20-08-36'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-29-20-16-55'

    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-20-27-48'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-20-36-43'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-29-20-45-38'

    # BertAligner

    # P = 0
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-30-10-13-10'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-30-10-21-58'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0_DROP0_ME3_2024-03-30-10-28-55'

    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-30-10-42-49'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-30-10-49-39'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_EW-TT-MT_en-it_context_fix_TS_U0_S1_ING_P0.1_DROP0_ME3_2024-03-30-10-56-35'

    # mbert on xl-wa
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_mbert_xlwa_en-it_ME3_2024-03-30-11-03-31'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased/bert-base-multilingual-cased_mbert_xlwa_en-it_ME3_2024-03-30-11-11-09'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_mbert_xlwa_en-it_ME3_2024-03-30-11-22-29' # no dropout
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_mbert_xlwa_en-it_ME3_2024-03-30-11-30-02'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_mbert_xlwa_en-it_ME2_2024-03-30-11-35-41'
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_BertAligner/bert-base-multilingual-cased_BertAligner_mbert_xlwa_en-it_ME3_2024-03-30-11-41-41'

    # mbert_xl-wa_en-it-es on ew-tt-mt_en-it-es
    # P = 0
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_mbert_xlwa_en-it-es_ME3_2024-03-30-16-09-15/bert-base-multilingual-cased_mbert_xlwa_en-it-es_ME3_2024-03-30-16-09-15_EW-TT-MT_multi_context_TS_U0_S1_ING_P0_DROP0_en-it-es_ME3_2024-03-30-17-17-27'
    # P = 0.1
    # model_name = '/home/pgajo/food/models/alignment/bert-base-multilingual-cased_mbert_xlwa_en-it-es_ME3_2024-03-30-16-09-15/bert-base-multilingual-cased_mbert_xlwa_en-it-es_ME3_2024-03-30-16-09-15_EW-TT-MT_multi_context_TS_U0_S1_ING_P0.1_DROP0_en-it-es_ME3_2024-03-30-17-39-01'

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
        
        evaluator.get_eval_batch(outputs, batch, split, type_labels = False)

    evaluator.evaluate(model, split, epoch, eval_metric='test')
    epoch_test_loss /= len(dataset_test[split])
    evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

    evaluator.print_metrics(current_epoch = epoch, current_split = split)
    evaluator.store_metrics()

    results_path = '/home/pgajo/food/results/alignment/test'
    # model save folder
    model_name = model_name.split('/')[-1]
    data_name = data_repo_test.split('/')[-1]
    data_results_path = os.path.join(results_path, data_name)
    if not os.path.isdir(data_results_path):
        os.makedirs(data_results_path)
    save_name = f"{model_name}_E{evaluator.epoch_best}_{split.upper()}{str(round(evaluator.exact_dev_best, ndigits=0))}"
    save_name = save_name.replace('bert-base-multilingual-cased', 'mbert')
    save_name = save_name.replace('mdeberta-v3-base', 'mdeberta')
    metrics_save_path = os.path.join(data_results_path, save_name)
    evaluator.save_metrics_to_csv(metrics_save_path)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()