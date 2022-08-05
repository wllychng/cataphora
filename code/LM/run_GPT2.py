
# 6/30/22 edit/copy, runs stimuli through GPT2 model, using pandas to access data 

# new version of running GPT2 to get loss from stimuli
# does not use CataphoraDataset class (subclass of HF Dataset);
# instead, simply uses a dictionary to store/access the stimuli data 

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from pathlib import Path
# import csv
# from load_tsv import load_tsv, DataFileType
import pandas as pd
# from numpy import nan

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
#MODEL = "gpt2-medium"
#MODEL = "gpt2"
MODEL = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(MODEL).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL)

DATA_DIR = "/home/cheung.179/cataphora_data"
DATASETS = [Path(DATA_DIR, "Kazanina07_direct.tsv")]

df = pd.read_csv(DATASETS[0], sep="\t")
for i in df.index:
	print(df.loc[i])

df[] = nan*df.shape[0]

# make a new df that contains minimal pair	

exit()

# don't write tsv file manually; make/edit pandas df, then use df.to_csv() to write at the end

# delete rows that are missing a set_id value


OUT_TSV_FILE = Path(DATA_DIR, MODEL+"_output.tsv")

# make list with set_ids to iterate over
X = df["set_id"].unique()
print(X)
exit()

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

