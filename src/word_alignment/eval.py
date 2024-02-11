import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset
import pandas as pd

def main():
    # model_repo = 'pgajo/mbert_TASTEset_U1_S0_DROP1_E10_DEV92.0'
    # model_repo = 'pgajo/mbert_TASTEset_U0_S1_DROP1_E10_DEV80.0'
    # model_repo = 'pgajo/mdeberta_TASTEset_U1_S0_DROP1_E1_DEV98.0'
    model_repo = 'pgajo/mdeberta_TASTEset_U0_S1_DROP1_E1_DEV89.0'
    # tokenizer_name = 'microsoft/mdeberta-v3-base'
    # model_repo = 'microsoft/mdeberta-v3-base'
    
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

    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    # data_repo = 'pgajo/mbert_GZ-GOLD-NER-ALIGN_105_U1_S0_DROP0'
    data_repo = 'pgajo/mdeberta_GZ-GOLD-NER-ALIGN_105_U1_S0_DROP0'
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
    results_path = '/home/pgajo/working/food/results'
    model_results_path = os.path.join(results_path, data_repo.split('/')[-1])
    evaluator.save_metrics_to_csv(os.path.join(model_results_path, model_repo.split('/')[-1]))

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()