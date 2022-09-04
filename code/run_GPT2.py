
# 8/26/22 copy: process stimuli through GPT2, use pandas to access data

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import pandas as pd
from pathlib import Path
import argparse
from stimuli_utils import clean_stimuli, make_output_filename

argParser = argparse.ArgumentParser()
argParser.add_argument("--dataset", default="../data/test_set1.tsv")
argParser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large"])
args = argParser.parse_args()

print(f"dataset: {args.dataset}")
print(f"model: {args.model}")

# set up neural model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}")
model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained(args.model)

df = pd.read_csv(Path(args.dataset), sep="\t")
df = clean_stimuli(df)
#print(df)

high_loss = [] 
low_loss = []

for row in df.itertuples():
    # extract necessary values from the df row
	# need the text example
	low_loss_ex = row.low_loss_example
	high_loss_ex = row.high_loss_example
	print(f"high loss example: {high_loss_ex}")
	print(f"low loss example: {low_loss_ex}")

	# interface with neural model
	low_encodings = tokenizer(low_loss_ex, return_tensors="pt")
	low_input_ids = low_encodings.input_ids.to(DEVICE)
	with torch.no_grad():
		low_outputs = model(low_input_ids, labels=low_input_ids.clone())
	low_loss.append(low_outputs.loss.item())
	
	high_encodings = tokenizer(high_loss_ex, return_tensors="pt")
	high_input_ids = high_encodings.input_ids.to(DEVICE)
	with torch.no_grad():
		high_outputs = model(high_input_ids, labels = high_input_ids.clone())
	# print(f"HF loss: {outputs.loss.item()}")
	high_loss.append(high_outputs.loss.item())

df["high_loss"] = high_loss
df["low_loss"] = low_loss

df["loss_diff"] = df["high_loss"] - df["low_loss"]

out_filename = make_output_filename(args.dataset, "lossNN")
print(f"out file: {out_filename}")

df.to_csv(out_filename, sep = "\t")
