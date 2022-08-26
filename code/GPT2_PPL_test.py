from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from cataphora_data import CataphoraDataset
from pathlib import Path
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
#MODEL = "gpt2-medium"
MODEL = "gpt2"
#MODEL = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(MODEL).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL)


# the perplexity of correct sentence should be lower than scrambled words
# the perplexity of subj-verb agreement sentence should be lower than non-agreement

DATA_DIR = "/home/cheung.179/cataphora_data"
# DATASETS = [Path(DATA_DIR, "Kazanina07.tsv"), Path(DATA_DIR, "GordonHendrick.tsv"), Path(DATA_DIR, "WebNLG.tsv")]
DATASETS = [Path(DATA_DIR, "Kazanina07.tsv")]

data = CataphoraDataset(DATASETS)

OUT_TSV_FILE = Path(DATA_DIR, MODEL+"_output.tsv")

with open(OUT_TSV_FILE, 'w') as out_file:
	# make a new tsv with just the example, id, and loss
	out_fieldnames = ["example", "set_id", "condition_id", "loss"]
	out_tsv = csv.DictWriter(out_file, dialect="excel-tab", fieldnames=out_fieldnames)
	out_tsv.writeheader()

	for x in data:
		text = x["example"]	
		set_id = x["set_id"]
		condition_id = x["condition_id"]
		encodings = tokenizer(text, return_tensors="pt")
		input_ids = encodings.input_ids.to(device)
		target_ids = input_ids.clone()
		with torch.no_grad():
			outputs = model(input_ids, labels=target_ids)
		print(f"text: {text} , set_id: {set_id} , condition_id: {condition_id} , loss: {outputs.loss.item()}")
		out_tsv.writerow({"example":text, "set_id":set_id, "condition_id":condition_id, "loss":outputs.loss.item()})
		#perplexity = torch.exp(outputs.loss)
		#print(f"PPL: {perplexity}")

