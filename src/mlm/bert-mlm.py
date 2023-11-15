from transformers import BertConfig, BertModel

# Define the configuration
config = BertConfig(
  vocab_size=1000,
  hidden_size=768,
  num_attention_heads=12,
  num_hidden_layers=12,
  intermediate_size=3072,
  hidden_dropout_prob=0.1,
  attention_probs_dropout_prob=0.1
)

# Instantiate model
model = BertModel(config)
