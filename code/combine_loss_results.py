
# combines all .tsv files from different models for single given dataset
# combines all file that match pattern
# seems outdated

# TODO: 12/6 update to better figure out files to combine

import logging

import pandas
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)

DIR = Path("../data")
SET = "HarrisBates2002"

files = os.listdir(DIR)

re_pattern = SET + ".*lossNN.*tsv"
files_to_combine = [x for x in files if re.match(re_pattern, x) != None]
logging.info(f" files to combine:\n{files_to_combine}")

# pattern for finding model name from file
# using ? after quantifier * means it will do non-greedy match, stopping at first _ reached
re_model = ".*_(gpt2.*?)_.*"

# start with empty df
df = pandas.DataFrame()

# iterate through dfs from files
# on first iteration (df is empty), set df equal to input
# on subsequent iterations, concat columnwise (axis=1)
for file in files_to_combine:
    cur_df = pandas.read_csv(DIR / file, sep="\t")
    model_match = re.match(re_model, file)
    model = model_match.group(1)
    logging.info(f" model name from regex: {model}")
    if df.empty:
        # initialize df with non-loss columns
        df = cur_df[['high_loss_example', 'low_loss_example', 'set_id', 'cataphora_condition']]
    cur_df = cur_df.select_dtypes(include='float')
    # add model as prefix to column labels
    cur_df = cur_df.add_prefix(model+":")
    df = pandas.concat([df, cur_df], axis=1)

logging.info(f" df:\n{df}")
df.info()

out_file = DIR / (SET + "_" + "lossAll" + ".tsv")
logging.info(f" writing to file: {out_file}")
df.to_csv(out_file, sep="\t")

