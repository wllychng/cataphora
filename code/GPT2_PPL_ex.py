from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# X = encodings.input_ids
# print(f"type of input ids: {type(X)}")
# print(f"size of input ids: {X.size()}")

max_length = model.config.n_positions
print(f"max len of model: {max_length}, type: {type(max_length)}")
# stride = 512
stride = 5

SHORT = 100
input_ids = encodings.input_ids[:, :SHORT]
print(f"input ids: {input_ids}")

nlls = []
# input_ids.size(1) is the sequence length
# i increments by stride every step
for i in range(0, input_ids.size(1), stride):
	# i + stride represents the end index. 
	# - max_length then gives us start index with maximal amount of context available
	begin_loc = max(i + stride - max_length, 0)
	end_loc = min(i + stride, encodings.input_ids.size(1))
	trg_len = end_loc - i
	print(f"begin_loc: {begin_loc}, end_loc: {end_loc}, trg_len: {trg_len}")
	temp_input_ids = input_ids[:, begin_loc:end_loc].to(device)
	target_ids = temp_input_ids.clone()
	# negative indexing means get everything except trg_len number of elements at the end
	# target ids set to -100 won't affect PPL
	target_ids[:, :-trg_len] = -100	 
	# print(f"encoding input ids: {temp_input_ids}")
	# print(f"target label ids: {target_ids}")

	with torch.no_grad():
		outputs = model(temp_input_ids, labels=target_ids)
		print(f"i: {i} , loss: {outputs.loss}")
		# seems like the loss is the average loss over the entire target sequence\
		# multiply by trg_len to get total loss of target sequence
		neg_log_like = outputs.loss * trg_len
		print(f"neg log like: {neg_log_like}")

	nlls.append(neg_log_like)

print(f"final end index: {end_loc}")
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

print(f"ppl: {ppl}")
