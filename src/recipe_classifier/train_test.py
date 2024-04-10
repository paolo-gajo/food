from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import torch

def main():
    # model_name = "microsoft/mdeberta-v3-base"
    # model_name = 'microsoft/deberta-v3-base'
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = load_from_disk('/home/pgajo/food/datasets/classification/food-com-3-diets')
    
    nrows = None
    data['train'] = data['train'].select(range(len(data['train']))[:nrows])
    data['dev'] = data['dev'].select(range(len(data['dev']))[:nrows])
    data['test'] = data['test'].select(range(len(data['test']))[:nrows])

    training_label = 'kosher'
    # training_label = 'vegetarian'
    # training_label = 'vegan'

    training_text = 'name_ingredients'
    # training_text = 'full_text'

    def preprocess_function(examples):
        inputs = tokenizer(
                            examples[training_text],
                            truncation=True,
                            padding='longest',
                            return_tensors='pt'
                            )
        inputs.update({'labels': torch.tensor(examples[training_label])})
        return inputs
    
    batch_size = 32
    data = data.map(preprocess_function, batched=True, batch_size=None)
    print(data)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = f1_metric.compute(predictions=predictions, references=labels)
        return metrics

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_name_simple = model_name.split('/')[-1]

    epochs = 1
    
    model_dir = f'/home/pgajo/food/models/classification/{model_name_simple}/{training_label}_ME{epochs}_{training_text}_nrows{nrows}'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics["train_samples"] = len(data["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(data["test"], metric_key_prefix="predict")
    
    actual_predictions = [np.argmax(pred, axis=0) for pred in predictions]

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    # Save predictions
    output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predictions_file, "w") as writer:
            for prediction in actual_predictions:
                writer.write(str(prediction))
    # overall_f1s.append(metrics['predict_overall_f1'])
    # ic(overall_f1s)
    # ic(np.mean(overall_f1s))
    os.rename(model_dir, model_dir + f"_{round(metrics['predict_f1'], 2)}")

if __name__ == '__main__':
    main()