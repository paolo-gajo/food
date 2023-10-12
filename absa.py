from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")  # replace with your model name
model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")  # replace with your model name

# Sentence and target word
sentence = "Consuming butter increases cholesterol in blood."
target_word = "butter"

# Prepare the text input, note the [CLS] and [SEP] tokens 
text = f"[CLS] {sentence} [SEP] {target_word} [SEP]"

# Encode the text as input for BERT
inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=False)  # The special tokens are already added manually

# Get model's output
outputs = model(**inputs)

# The output is a tuple, with the logits as the first item
logits = outputs[0]
print(logits)

# Get the predicted sentiment by choosing the highest logit
predicted_sentiment = logits.argmax().item()

# Translate the prediction to a sentiment label
sentiments = ['negative', 'neutral', 'positive']
predicted_label = sentiments[predicted_sentiment]

print(f"The sentiment of the word '{target_word}' in the given sentence is: {predicted_label}")
