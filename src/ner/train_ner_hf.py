from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from icecream import ic
import os
import re
import pandas as pd
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import data_loader
# sys.path.append('/home/pgajo/food/bert_crf')
sys.path.append('/home/pgajo/food/data/TASTEset-2.0/src')
from BERT_with_CRF import BERTCRF
import torch
from tqdm.auto import tqdm
import evaluate
import argparse

login(os.environ['HF_TOKEN'])
from ner_utils import make_ner_sample, get_ner_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('--model')
    parser.add_argument('--langs_train')
    parser.add_argument('--langs_test')
    args = parser.parse_args()

    print('Current dataset:', args.train_path)
    train_name_simple = args.train_path.split('/')[-1]
    test_name_simple = args.test_path.split('/')[-1]
    dataset = load_from_disk(args.train_path)

    print(dataset)
    label_field = 'annotations'
    raw_labels, label_list, label2id, id2label = get_ner_classes(dataset, label_field=label_field)
    ic(raw_labels)
    ic(label_list)

    from transformers import AutoTokenizer
    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    languages_train = list(args.langs_train.split('-'))
    languages_test = list(args.langs_test.split('-'))

    df_train = pd.DataFrame()
    df_dev = pd.DataFrame()

    for lang in languages_train:
        df_train = pd.concat([df_train, pd.DataFrame([make_ner_sample(sample,
                                                        tokenizer,
                                                        label2id,
                                                        lang,
                                                        label_list=raw_labels,
                                                        label_field=label_field,
                                                        text_name = 'ingredients') for sample in dataset['train']])],
                                                        )
        df_dev = pd.concat([df_dev, pd.DataFrame([make_ner_sample(sample,
                                                        tokenizer,
                                                        label2id,
                                                        lang,
                                                        label_list=raw_labels,
                                                        label_field=label_field,
                                                        text_name = 'ingredients') for sample in dataset['test']])],
                                                        )

    dataset_train = Dataset.from_pandas(df_train)
    dataset_dev = Dataset.from_pandas(df_dev)

    dataset_traindev = DatasetDict()
    dataset_traindev['train'] = dataset_train
    dataset_traindev['dev'] = dataset_dev
    ic(dataset_traindev)

    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    import numpy as np

    seqeval = evaluate.load("seqeval")

    from sklearn.metrics import f1_score

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        micro_f1 = f1_score(flat_true_labels, flat_true_predictions, average='micro')
        flat_results = {}
        for key in results.keys():
            if isinstance(results[key], dict):
                for key_inner in results[key].keys():
                    flat_results[key+'_'+key_inner] = results[key][key_inner]
            else:
                flat_results[key] = results[key]
        flat_results['micro_f1'] = micro_f1
        ic(flat_results)
        return flat_results

    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model_name_simple = model_name.split('/')[-1]
    crf = 0
    
    if crf:
        model = BERTCRF.from_pretrained(
                                        model_name,
                                        num_labels=len(label_list),
                                        # id2label=id2label,
                                        # label2id=label2id
                                    )
        model_name = model_name + '_CRF'
    else:
        model = AutoModelForTokenClassification.from_pretrained(
                                                        model_name,
                                                        num_labels=len(label_list),
                                                        # id2label=id2label,
                                                        # label2id=label2id
                                                    )
    ic(model_name)

    from datetime import datetime
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = os.path.join('/home/pgajo/food/models/ner',
                                f"{'-'.join(languages_train)}_train_{'-'.join(languages_test)}_test",
                                f"model={model_name_simple}",
                                f"train={train_name_simple}",
                                f"test={test_name_simple}",
                                date_time)

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        # load_best_model_at_end=True,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_traindev["train"],
        eval_dataset=dataset_traindev["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics["train_samples"] = len(dataset_traindev)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    data_name_test_simple = args.test_path.split('/')[-1]
    dataset_test_raw = load_from_disk(args.test_path)
    label_field = 'annotations'
    # print(dataset_test_raw['train'][label_field])
    # label_list, label2id, id2label = get_ner_classes(dataset_test_raw, label_field=label_field)
    # ic(label_list)
    ic(len(dataset_test_raw['train']))
    df_test = pd.DataFrame()

    verbose = 0
    for lang in languages_test:
        sample_buffer = []
        for i, sample in enumerate(dataset_test_raw['train']):
            # if i > 301:
            #     break
            ner_sample = make_ner_sample(sample,
                                        tokenizer,
                                        label2id,
                                        text_name='ingredients',
                                        label_field=label_field,
                                        lang=lang,
                                        label_list=raw_labels)
            # if len(ner_sample['labels']) > 512:
            if verbose:
                # print('text', tokenizer.decode(ner_sample['input_ids']))
                # print('input_ids', ner_sample['input_ids'])
                # print('decoded ids', [tokenizer.decode(id) for id in ner_sample['input_ids']])
                # print('labels', ner_sample['labels'])
                print('line number:', i)
                for id, label in zip(ner_sample['input_ids'], ner_sample['labels']):
                    if id != tokenizer.pad_token_id:
                        print(id, tokenizer.decode(id), label, (id2label[label] if label in id2label.keys() else 'None'), sep = '\t')
                print('##################################################################')
                # raise Exception("Length > 512")
            sample_buffer.append(ner_sample)
        df_test = pd.concat([df_test, pd.DataFrame(sample_buffer)])
    print(len(df_test))
    dataset_test = Dataset.from_pandas(df_test)

    dataset_test_dict = DatasetDict()
    dataset_test_dict['test'] = dataset_test
    dataset_test_dict.set_format('torch')
    ic(dataset_test_dict)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(dataset_test, metric_key_prefix="predict")
    
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    # Save predictions
    output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")
    # overall_f1s.append(metrics['predict_overall_f1'])
    # ic(overall_f1s)
    # ic(np.mean(overall_f1s))
    os.rename(model_dir, model_dir + f"_{round(metrics['predict_overall_f1'], 2)}")
if __name__ == '__main__':
        main()