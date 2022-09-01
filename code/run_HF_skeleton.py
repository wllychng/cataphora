# skeleton code to show basic workflow of HuggingFace model (in particular, GPT2)

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# loss does not work with only a single token
# test_sentence = "X"
test_sentence = "hello world"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

encodings = tokenizer(test_sentence, return_tensors="pt")
input_ids = encodings.input_ids.to(DEVICE)
target_ids = input_ids.clone()
with torch.no_grad():
    outputs = model(input_ids, labels=target_ids)
print(f"HF outputs: {outputs}")
print(f"type of outputs: {type(outputs)}")
print(f"loss: {outputs.loss.item()}")
