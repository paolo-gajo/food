import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import save_local_model
import os

class TranslationDataset(Dataset):
    def __init__(self, text_pairs, tokenizer, max_length=128):
        self.text_pairs = text_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text_pairs)
    
    def __getitem__(self, idx):
        en_text, it_text = self.text_pairs[idx]
        en_tokens = self.tokenizer(en_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        it_tokens = self.tokenizer(it_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return en_tokens, it_tokens

def train(model, device, data_loader, optimizer):
    model.train()
    total_loss = 0
    for en_tokens, it_tokens in data_loader:
        optimizer.zero_grad()
        
        en_tokens = {key: val.squeeze().to(device) for key, val in en_tokens.items()}
        it_tokens = {key: val.squeeze().to_device for key, val in it_tokens.items()}
        
        with torch.no_grad():
            en_output = model(**en_tokens).pooler_output
            it_output = model(**it_tokens).pooler_output
        
        loss = -cosine_similarity(en_output, it_output).mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    model_name = 'bert-base-multilingual-cased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained()
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    
    # Example data: list of tuples (English sentence, Italian sentence)
    text_pairs = [
        ("house", "casa"),
        ("apple", "mela"),
        ("book", "libro")
    ]
    
    dataset = TranslationDataset(text_pairs, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    num_epochs = 3
    for epoch in range(num_epochs):
        loss = train(model, device, data_loader, optimizer)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    model_dir = '/home/pgajo/food/models/embeddings'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_save_path = os.path.join(model_dir, model_name.split('/')[-1])
    save_local_model(model_save_path, model, tokenizer)

if __name__ == "__main__":
    main()
