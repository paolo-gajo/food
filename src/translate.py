# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-it')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-it')

# tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-it')
# model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-it')

# model.get_encoder().to('cuda:0')
# model.get_decoder().to('cuda:1')
# print(model.config)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

input_text = """
1 cup chocolate chips;1 cup heavy cream;1 egg;whipped cream;1 teaspoon vanilla extract;4 ripe coconuts;1 cup evaporated milk;1 cup gin;3 tablespoons sugar (optional);1 teaspoon ground cinnamon;1/2 teaspoon freshly grated nutmeg
"""
batch = tokenizer([input_text], return_tensors="pt")
print(type(batch))
generated_ids = model.generate(**batch, max_new_tokens = 1024)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])




