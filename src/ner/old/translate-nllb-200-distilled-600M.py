from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# model.get_encoder().to('cuda:0')
# model.get_decoder().to('cuda:1')
# print(model.config)

input_text = """
1 cup chocolate chips;1 cup heavy cream;1 egg;whipped cream;1 teaspoon vanilla extract;4 ripe coconuts;1 cup evaporated milk;1 cup gin;3 tablespoons sugar (optional);1 teaspoon ground cinnamon;1/2 teaspoon freshly grated nutmeg
"""
inputs = tokenizer(input_text, return_tensors="pt")

# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200 for lang codes
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"], max_length=256
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
