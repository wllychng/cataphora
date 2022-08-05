# new version of running GPT2 to get loss from stimuli
# does not use CataphoraDataset class (subclass of HF Dataset);
# instead, simply uses a dictionary to store/access the stimuli data 

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from pathlib import Path
import csv
from load_tsv import load_tsv, DataFileType

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
#MODEL = "gpt2-medium"
#MODEL = "gpt2"
MODEL = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(MODEL).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL)

# the perplexity of correct sentence should be lower than scrambled words
# the perplexity of subj-verb agreement sentence should be lower than non-agreement

DATA_DIR = "/home/cheung.179/cataphora_data"
DATASETS = [Path(DATA_DIR, "Kazanina07_direct.tsv")]

data = load_tsv(DATASETS[0], DataFileType.DIRECT)

OUT_TSV_FILE = Path(DATA_DIR, MODEL+"_output.tsv")

with open(OUT_TSV_FILE, 'w') as out_file:
	# make a new tsv with just the example, id, and loss
	out_fieldnames = ["example", "set_id", "condition_id", "loss"]
	out_tsv = csv.DictWriter(out_file, dialect="excel-tab", fieldnames=out_fieldnames)
	out_tsv.writeheader()

	for set_id, cond_dict in data.items():
		# data is a dictionary, x is a string key of the dictionary
		print(f"set_id: {set_id}")
		for cond, example in cond_dict.items():
			print(f"condition: {cond}")
			print(f"example: {example}")
		
			encodings = tokenizer(example, return_tensors="pt")
			input_ids = encodings.input_ids.to(device)
			target_ids = input_ids.clone()
			with torch.no_grad():
				outputs = model(input_ids, labels=target_ids)
			loss = outputs.loss.item()
			print(f"loss: {loss}")
			out_tsv.writerow({"example":example, "set_id":set_id, "condition_id":cond, "loss":loss})
			#perplexity = torch.exp(outputs.loss)
			#print(f"PPL: {perplexity}")

