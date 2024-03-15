import warnings
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, BertModel, PretrainedConfig, BertPreTrainedModel
from utils import data_loader, SquadEvaluator, TASTEset, save_local_model
from aligner_pt import BertAligner
import re
from datetime import datetime

import sys
sys.path.append('/home/pgajo/food/bert_crf')
from crf_models.bert_crf import BertCrf

def main():
    model_repo = 'bert-base-multilingual-cased'

    # data_repo = '/home/pgajo/food/datasets/EW-TT-MT_LOC_en-it_U0_S1_Tingredient_P0.25_DROP1_mbert'
    # data_repo = '/home/pgajo/food/datasets/EW-TT-MT_LOC_en-it_U1_S0_Trecipe_P0_DROP1_mbert' # unshuffled MT-LOC
    data_repo = '/home/pgajo/food/datasets/EW-TT-PE_en-it_U0_S1_Tingredient_P0.25_DROP1_mbert'
    # data_repo = '/home/pgajo/food/datasets/EW-TT-PE_en-it_U1_S0_Trecipe_P0_DROP1_mbert' # unshuffled PE

    data = TASTEset.from_disk(data_repo)
    batch_size = 64
    dataset = data_loader(data,
                        batch_size,
                        # n_rows=320,
                        )
    device = 'cuda'
    model = BertAligner.from_pretrained(model_repo,
                                        )#.to(device)
    # model = BertCrf(2, model_repo)
    
    model = torch.nn.DataParallel(model).to(device)
    
    lr = 3e-5
    eps = 1e-8
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                lr = lr,
                                eps = eps
                                )
    
    tokenizer = BertTokenizer.from_pretrained(model_repo)
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
            loss = outputs['loss'].mean()
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
            input = {k: batch[k].to(device) for k in columns}
            with torch.inference_mode():
                outputs = model(**input)
            loss = outputs['loss'].mean()
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

    # results_path = '/home/pgajo/food/results'
    # # model save folder
    # model_dir = './models'
    # model_name = model_repo.split('/')[-1].split('_')[0]
    # data_name = re.sub('.json', '', data_repo.split('/')[-1]) # remove extension if local path
    # data_results_path = os.path.join(results_path, data_name)
    # if not os.path.isdir(data_results_path):
    #     os.makedirs(data_results_path)
    # date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # save_name = f"{model_name}_{data_name}_E{evaluator.epoch_best}_DEV{str(round(evaluator.exact_dev_best, ndigits=0))}_{date_time}"
    # save_name = save_name.replace('bert-base-multilingual-cased', 'mbert')
    # save_name = save_name.replace('mdeberta-v3-base', 'mdeberta')
    # csv_save_path = os.path.join(data_results_path, save_name)
    # print('Saving metrics to:', csv_save_path)
    # evaluator.save_metrics_to_csv(csv_save_path)

    # model_save_dir = os.path.join(model_dir, f"{data.name}/{save_name}")
    # if not os.path.isdir(model_save_dir):
    #     os.makedirs(model_save_dir)
    # evaluator.save_metrics_to_csv(os.path.join(model_save_dir, 'metrics'))
    # save_local_model(model_save_dir, model, tokenizer)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
