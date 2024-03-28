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

def main():

    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base_EW-TT-PE_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mdeberta_align_E6_DEV95.0_20240326-10-54-48'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base_EW-TT-MT_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mdeberta_align_E3_DEV98.0_2024-03-26-12-21-37'

    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it_EW-TT-MT_LOC_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mdeberta_align_E3_DEV95.0_2024-03-26-14-21-06'

    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it_EW-TT-MT_en-it_spaced_TS_U0_S1_ING_P1_DROP1_mdeberta_align_E10_DEV98.0_2024-03-26-21-34-15'
    
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it'

    # model_name = 'pgajo/mbert-xlwa-en-it'
    # model_name = '/home/pgajo/food/models/alignment/mbert_EW-TT-MT_LOC_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mbert_align_E7_DEV81.0_2024-03-26-14-32-13'
    # model_name = '/home/pgajo/food/models/alignment/mbert_xlwa_en-it_EW-TT-MT_LOC_en-it_spaced_TS_U0_S1_ING_P0.5_DROP1_mbert_align_E5_DEV82.0_2024-03-26-14-53-28'

    # model_name = '/home/pgajo/food/models/alignment/mdeberta-v3-base_EW-TT-MT_en-it_context_U0_S1_ING_P0.5_DROP1_mdeberta_align_E3_DEV96.0_2024-03-27-10-07-45'
    model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_U0_S1_ING_P0.5_DROP1_mdeberta_align_E8_DEV98.0_2024-03-27-10-14-18'
    # model_name = '/home/pgajo/food/models/alignment/mdeberta_xlwa_en-it_EW-TT-MT_en-it_context_U0_S1_ING_P0.5_DROP1_mdeberta_align_E10_DEV97.0_2024-03-27-10-40-48_BO1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data_repo_test = 'pgajo/xlwa_en-it_mbert'
    # data_repo_test = 'pgajo/xlwa_en-it_mdeberta'
    # data_repo_test = '/home/pgajo/food/datasets/alignment/GZ-GOLD-NER-ALIGN_105_spaced_U1_S0_Trecipe_P0_DROP0_mbert_en-it_INV1_align'
    data_repo_test = '/home/pgajo/food/datasets/alignment/GZ-GOLD-NER-ALIGN_105_spaced_U1_S0_Trecipe_P0_DROP0_mdeberta_en-it_INV1_align'
    data_test = TASTEset.from_disk(data_repo_test)

    batch_size = 32
    dataset_test = data_loader(data_test, 
                        batch_size,
                        # n_rows=100
                        )

    device = 'cuda'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    # model = BertAligner.from_pretrained(model_name)
    # print(model.eval())
    
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

    evaluator.evaluate(model, split, epoch, eval_metric='test')
    epoch_test_loss /= len(dataset_test[split])
    evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

    evaluator.print_metrics(current_epoch = epoch, current_split = split)
    evaluator.store_metrics()

    results_path = '/home/pgajo/food/results/alignment'
    # model save folder
    model_name = model_name.split('/')[-1]
    data_name = re.sub('\..*', '', data_repo_test.split('/')[-1]) # remove extension if local path
    data_results_path = os.path.join(results_path, data_name)
    if not os.path.isdir(data_results_path):
        os.makedirs(data_results_path)
    save_name = f"{model_name}_{data_name}_E{evaluator.epoch_best}_{split.upper()}{str(round(evaluator.exact_dev_best, ndigits=0))}"
    save_name = save_name.replace('bert-base-multilingual-cased', 'mbert')
    save_name = save_name.replace('mdeberta-v3-base', 'mdeberta')
    metrics_save_path = os.path.join(data_results_path, save_name)
    evaluator.save_metrics_to_csv(metrics_save_path)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()