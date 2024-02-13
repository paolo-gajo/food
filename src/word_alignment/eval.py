import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset
import pandas as pd
import re

def main():
    # model_repo = 'pgajo/mbert_EW-TT-PE_U1_S0_DROP1_mbert_E10_DEV98.0'                                     # 0% recipe shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U1_S0_DROP1_E10_DEV98.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U1_S0_DROP1_mbert_E8_DEV98.0'                           # 0% recipe shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U1_S0_DROP1_E8_DEV98.0'
    # model_repo = 'pgajo/mdeberta-v3-base_EW-TT-PE_U1_S0_DROP1_mdeberta_E2_DEV100.0'                       # 0% recipe shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U1_S0_DROP1_E2_DEV100.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U1_S0_DROP1_mdeberta_E2_DEV100.0'                    # 0% recipe shuffle mdeberta xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U1_S0_DROP1_E2_DEV100.0'

    # model_repo = 'pgajo/mbert_EW-TT-PE_U0_S1_DROP1_mbert_E10_DEV72.0'                                     # 100% recipe shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U0_S1_DROP1_E10_DEV72.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U0_S1_DROP1_mbert_E9_DEV76.0'                           # 100% recipe shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U0_S1_DROP1_E9_DEV76.0'
    # model_repo = 'pgajo/mdeberta-v3-base_mdeberta_EW-TT-PE_U0_S1_DROP1_E8_DEV93.0'                        # 100% recipe shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U0_S1_DROP1_E8_DEV93.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_mdeberta_EW-TT-PE_U0_S1_DROP1_E8_DEV92.0'                     # 100% recipe shuffle mdeberta xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_DROP1_E8_DEV92.0'

    # model_repo = 'pgajo/mbert_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_E9_DEV76.0'                       # 100% ingredient shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_mbert_E9_DEV76.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P1_DROP16_DEV85.0'            # 100% ingredient shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_E6_DEV85.0'
    # model_repo = 'pgajo/mdeberta_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_mdeberta_E4_DEV96.0'                 # 100% ingredient shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_E4_DEV96.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_mdeberta_E5_DEV96.0'      # 100% ingredient shuffle mdeberta xlwa * --> 61 exact match!
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P1_DROP1_E5_DEV96.0'


    # model_repo = 'pgajo/mbert_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_mbert_E9_DEV89.0'                    # 75% ingredient shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_E9_DEV89.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_mbert_E7_DEV92.0'         # 75% ingredient shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_E7_DEV92.0'
    # model_repo = 'pgajo/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_mdeberta_E5_DEV98.0'              # 75% ingredient shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_E5_DEV98.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_mdeberta_E4_DEV98.0'   # 75% ingredient shuffle mdeberta xlwa *
    model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.75_DROP1_E4_DEV98.0'

    # model_repo = 'pgajo/mbert_EW-TT-PE_U0_S1_Tingredient_P0.5_DROP1_mbert_E8_DEV83.0'                     # 50% ingredient shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U0_S1_Tingredient_P0.50_E8_DEV83.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.5_DROP1_mbert_E6_DEV88.0'          # 50% ingredient shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.50_DROP1_E6_DEV88.0'
    # model_repo = 'pgajo/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.5_DROP1_mdeberta_E4_DEV97.0'               # 50% ingredient shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.50_DROP1_E4_DEV97.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.5_DROP1_mdeberta_E5_DEV96.0'    # 50% ingredient shuffle mdeberta xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.50_DROP1_E5_DEV96.0'

    # model_repo = 'pgajo/mbert_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_mbert_E10_DEV80.0'                   # 25% ingredient shuffle mbert base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_E10_DEV80.0'
    # model_repo = 'pgajo/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_mbert_E10_DEV87.0'        # 25% ingredient shuffle mbert xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mbert-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_E10_DEV87.0'
    # model_repo = 'pgajo/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_mdeberta_E9_DEV97.0'              # 25% ingredient shuffle mdeberta base *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_E9_DEV97.0'
    # model_repo = 'pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_mdeberta_E5_DEV95.0'   # 25% ingredient shuffle mdeberta xlwa *
    # model_repo = '/home/pgajo/working/food/models/TASTEset/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_E5_DEV95.0'

    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    # data_repo = 'pgajo/xlwa_en-it_mbert'
    # data_repo = 'pgajo/xlwa_en-it_mdeberta'
    # data_repo = 'pgajo/GZ-GOLD-NER-ALIGN_105_U1_S0_DROP0_mbert'
    data_repo = 'pgajo/GZ-GOLD-NER-ALIGN_105_U1_S0_DROP0_mdeberta'
    data = TASTEset.from_datasetdict(data_repo)

    batch_size = 32
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=100
                        )

    
    model = AutoModelForQuestionAnswering.from_pretrained(model_repo)
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
    
    evaluator.evaluate(model, split, epoch, eval_metric='test')
    epoch_test_loss /= len(dataset[split])
    evaluator.epoch_metrics[f'{split}_loss'] = epoch_test_loss

    evaluator.print_metrics(current_epoch = epoch, current_split = split)
    evaluator.store_metrics()

    results_path = '/home/pgajo/working/food/results_new'
    # model save folder
    model_name = model_repo.split('/')[-1]
    data_name = re.sub('\..*', '', data_repo.split('/')[-1]) # remove extension if local path
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