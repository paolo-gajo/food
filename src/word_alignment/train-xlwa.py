import os
import torch
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
import pandas as pd
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, push_model

lang_list = [
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

lang_id = '-'.join(lang_list)
data_path = f'/home/pgajo/working/food/data/XL-WA/data/.ready/{lang_id}'
results_path = f'/home/pgajo/working/food/src/word_alignment/XL-WA/results/{lang_id}'

data = DatasetDict.load_from_disk(data_path) # load prepared tokenized dataset
batch_size = 32
dataset = data_loader(data,
                      batch_size,
                      n_rows = 64,
                      )

model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = 'cuda'
model = torch.nn.DataParallel(model).to(device)

optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=2e-5,
                              eps=1e-8
                              )

evaluator = SquadEvaluator(tokenizer, 
                           model,
                           load("squad_v2"),
                           )

epochs = 3

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

    evaluator.evaluate(split, epoch)
    epoch_train_loss /= len(dataset[split])
    evaluator.metrics[epoch][f'{split}_loss'] = epoch_train_loss

    evaluator.print_metrics(epoch)

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
        epoch_train_loss += loss.item()
        loss_tmp = round(epoch_train_loss / (i + 1), 4)
        progbar.set_postfix({'Loss': loss_tmp})
        
        evaluator.get_eval_batch(outputs, batch, split)
    
    
    evaluator.evaluate(split, epoch)
    epoch_dev_loss /= len(dataset[split])
    evaluator.metrics[epoch][f'{split}_loss'] = epoch_dev_loss

    evaluator.print_metrics(epoch)

    if evaluator.stop_training:
        print(f'Early stopping triggered on epoch {epoch}.\nBest epoch: {evaluator.epoch_best}.')
    
evaluator.save_metrics_to_csv(results_path)

push_model(
    model,
    model_name,
    suffix = f'xl-wa-{lang_id}-{evaluator.exact_dev_best}',
    model_description = f'''
    Word alignment model fine-tuned on XL-WA ({lang_id})
    for {evaluator.epoch_best} epochs. \n\n
    Metrics:\n
    {pd.DataFrame(evaluator.metrics)}
    '''
)