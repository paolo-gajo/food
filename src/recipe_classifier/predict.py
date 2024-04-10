from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

def main():
    label = 'vegetarian'
    model_name = f"/home/pgajo/food/models/classification/bert-base-uncased/{label}_ME1_name_ingredients_nrowsNone_0.95"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = 'cuda')
    csv_path = '/home/pgajo/food/data/recipenlg/RecipeNLG_dataset.csv'
    data = pd.read_csv(csv_path)

    batch_size = 512
    preds = []
    with open(csv_path.replace('.csv', f'_{label}_preds.txt'), 'a', encoding='utf8') as f:
        with torch.inference_mode():
            for _, batch_data in tqdm(data.groupby(np.arange(len(data)) // batch_size), total=len(data)//batch_size):
                inputs = tokenizer([f"Name: {el['title']}\n\nIngredients: {', '.join(el['ingredients'])}" for _, el in batch_data.iterrows()],
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    )
                inputs = {k: inputs[k].to('cuda') for k in inputs.keys()}
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim = 1).tolist()
                for pred in batch_preds:
                    f.write(str(pred) + '\n')
                preds.extend(batch_preds)

    data[label] = preds

    data.to_csv(csv_path.replace('.csv', f'_{label}_preds.csv'))

if __name__ == '__main__':
    main()