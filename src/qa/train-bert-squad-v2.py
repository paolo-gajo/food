from datasets import load_dataset
import torch
import pickle

# dataset = load_dataset("squad_v2")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to('cuda')

# read from file
filehandler = open('./squad-v2-berttokenized.obj', 'rb') 
dataset = pickle.load(filehandler)

train = torch.utils.data.DataLoader(dataset['train'], batch_size = 1)
val = torch.utils.data.DataLoader(dataset['validation'], batch_size = 1)

for batch in train:
    print(batch.keys())
    # Assuming each key in the batch maps to a list of tensors
    input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
    print(input_ids)
    token_type_ids = torch.tensor(batch['token_type_ids']).unsqueeze(0)
    print(token_type_ids)
    attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0)
    print(attention_mask)

    # outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # print(outputs)
    # Rest of your code

    break  # Break the loop after the first batch