# script that takes a stimuli csv file in long format (a row for each item), and turns it into wide format, where each row represents two stimuli of minimal pair

# post-process data after running HuggingFace, 
# 1) set up a new df that holds minimal pair (MP) data
# 2) from original df, make a unique vector of set_id to iterate over
# 3) for each set_id, get the relevant rows/info, add a new row to the new df 

import pandas as pd
import argparse
from pathlib import Path
from stimuli_utils import make_output_filename

DEFAULT_FILE = "../data/sample_all.tsv"

argParser = argparse.ArgumentParser()
argParser.add_argument("data_file", nargs="?", default=None)
args = argParser.parse_args()

print(f"data file: {args.data_file}")

df = pd.read_csv(args.data_file, delimiter="\t")

# filter the df, removing rows with null set_id
df = df.loc[df["set_id"].notnull()]

# access column of values using indexing accessor [] 
set_ids = df["set_id"].unique()
# copy over the context to all the stimuli with the same set_id
for set_id in set_ids:
    # get a context for the set id; first filter df to only rows with current set id and non-null context, then pick out the context col, and first available row using iloc
    cur_context = df.loc[(df["set_id"] == set_id) & (df["context_1"].notnull()), "context_1"].iloc[0] 
    # get filtered df that only has non-null set_id and context
    # fill the set_id with empty contexts with cur_context
    df.loc[(df["set_id"]==set_id) & (df["context_1"].isnull()), "context_1"] = cur_context
    
# make a new dataframe that is long format, with the following columns
# col1: context
# col2: example
# col3: set_id
# col4: condition
# loss_1
# loss_2
# loss_diff

dict_for_df = {"context":[], "anaphora_example":[], "cataphora_example":[], "cataphora_condition":[], "set_id":[], "anaphora_loss":[], "cataphora_loss":[], "loss_diff":[]}

mp_df = pd.DataFrame({"context": df["context_1"], "anaphora_example":None, "cataphora_example":df["example"], "cataphora_condition":df["condition"], "set_id":df["set_id"], "anaphora_loss":None, "cataphora_loss":None, "loss_diff":None}, copy=True)

# use pathlib to make a path for the output tsv
file_path = Path(args.data_file)
output_file = file_path.parent / (file_path.stem + "_mp" + file_path.suffix)
print(f"output file: {output_file}")


