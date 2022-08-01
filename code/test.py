
# skeleton code to process stimuli
# fill in with NN processing later
# also change the name of the file

import pandas as pd
from pathlib import Path
import argparse

#DEFAULT_FILE = "../data/sample_all.tsv"
DEFAULT_FILE = "../data/test_dataset.xlsx"

argParser = argparse.ArgumentParser()
argParser.add_argument("data_file", nargs="?", default=DEFAULT_FILE)
args = argParser.parse_args()

print(f"data file: {args.data_file}")

df = pd.read_excel(args.data_file, sheet_name="test_set", usecols="A,D:G")

loss = [] 

for idx,row in df.iterrows():
    print(f"idx: {idx}")
    print(f'set_id: {row["set_id"]}')
    print(f'accept condition: {row["accept_condition"]}')

    loss.append(idx)

df["loss"] = loss

print(df)
# df.to_tsv()

file_path = Path(args.data_file)
out_filename = str(file_path.stem) + "_loss" + str(file_path.suffix)
print(f"out file: {out_filename}")
# out_file = file_path.parent + file_path.stem + "_loss" + file_path.suffix
