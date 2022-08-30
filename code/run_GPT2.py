
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

# df = pd.read_excel(args.data_file, sheet_name="test_set", usecols="A,D:G")
df = pd.read_csv(DATA_DIR / args.dataset, sep="\t")
print(df)

df = clean_stimuli(df)
print(df)

exit()

loss = [] 

for idx,row in df.iterrows():
    # extract necessary values from the df row
	# need the text example	
	temp_example = 1
	print(f"idx: {idx}")
	print(f'set_id: {row["set_id"]}')
	print(f'accept condition: {row["accept_condition"]}')

	# interface with neural model
	encodings = tokenizer(example, return_tensors="pt")
	input_ids = 1
	
	loss.append(idx)

df["loss"] = loss

print(df)
# df.to_tsv()

file_path = Path(args.data_file)
out_filename = str(file_path.stem) + "_loss" + str(file_path.suffix)
print(f"out file: {out_filename}")

# out_file = file_path.parent + file_path.stem + "_loss" + file_path.suffix
