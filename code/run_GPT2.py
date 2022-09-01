
# 8/26/22 copy: process stimuli through GPT2, use pandas to access data

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import pandas as pd
from pathlib import Path
import argparse
from stimuli_utils import clean_stimuli

DATA_DIR = Path("../data")

argParser = argparse.ArgumentParser()
argParser.add_argument("--dataset", default="sample_all.tsv")
argParser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large"])
args = argParser.parse_args()

print(f"dataset: {args.dataset}")
print(f"model: {args.model}")

# set up neural model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}")
model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained(args.model)

df = pd.read_csv(DATA_DIR / args.dataset, sep="\t")
print(df)

df = clean_stimuli(df)
print(df)

loss = [] 

for row in df.itertuples():
    # extract necessary values from the df row
	# need the text example
	temp_example = row.example
	print(f"temp_example: {temp_example}")
	print(f"row: {row}")	

	# interface with neural model
	encodings = tokenizer(temp_example, return_tensors="pt")
	input_ids = encodings.input_ids.to(DEVICE)
	target_ids = input_ids.clone()
	print(f"input_ids: {input_ids}")
	print(f"target_ids: {target_ids}")
	with torch.no_grad():
		outputs = model(input_ids, labels=target_ids)
	print(f"HF loss: {outputs.loss.item()}")
	# loss.append(outputs.loss.item())

df["loss"] = loss

print(df)
# df.to_tsv()

file_path = Path(args.data_file)
out_filename = str(file_path.stem) + "_loss" + str(file_path.suffix)
print(f"out file: {out_filename}")

# out_file = file_path.parent + file_path.stem + "_loss" + file_path.suffix
