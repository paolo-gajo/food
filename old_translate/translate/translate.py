# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-it')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-it')

# tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-it')
# model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-it')

# model.get_encoder().to('cuda:0')
# model.get_decoder().to('cuda:1')
# print(model.config)

# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters in the model: {num_params}")

def translate_marianmt(text):
    batch = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(**batch, max_new_tokens = 1024)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

input_text = """
6 tomatoes, cut into bite-size pieces;1/4 cup extra-virgin olive oil, or more to taste;1 1/2 tablespoons balsamic vinegar;6 leaves fresh basil, cut into slivers;1/2 pound mozzarella cheese, cut into bite-size cubes;salt and ground black pepper to taste

"""

print(translate_marianmt(input_text))



