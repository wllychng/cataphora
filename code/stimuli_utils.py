
# this file contains various functions for cleaning/processing the stimuli data

import pandas as pd
from pathlib import Path
from datetime import date

def clean_stimuli(df):
	# delete rows where stimuli is NaN
	assert "high_loss_example" in df.columns
	assert "low_loss_example"in df.columns
	df.dropna(subset = ["high_loss_example", "low_loss_example"], inplace = True)
	return df

# make sure dataset loaded into df conforms to expectations of stimuli data,
# throw errors otherwise
def validate_df(df):
	print("TODO")
	return True

# make a helpful output filename. takes two arguments, input file to base output file on, and an additional string identifier to concat.
def make_output_filename(input_path, str_identifier, output_dir=None):
	# make sure input_path is Path object, Path constructor is idempotent
	input_path = Path(input_path)
	# make output filename by taking the input stem, concat str_identifier, then concat 
	output_filename = "_".join([input_path.stem, str_identifier, str(date.today())+input_path.suffix])
	
	if output_dir == None: # use the same dir as the input file
		output_dir = input_path.parent
	
	return Path(output_dir, output_filename) 

if __name__ == "__main__":
    # test clean_stimuli function
    #test_df = pd.DataFrame({"col1": [1, 2, 3], "example": ["A", None, "C"]})
    #test_df = clean_stimuli(test_df)
    #print(test_df)

	# test make_output_filename
	INPUT_PATH = Path("a/b/c.txt")
	print(f"test input path: {INPUT_PATH}")
	STR_IDENTIFIER = "X"
	print(f"test string identifier: {STR_IDENTIFIER}")
	OUTPUT_DIR = "d/e"
	print(f"test output dir: {OUTPUT_DIR}")
	output = make_output_filename(INPUT_PATH, STR_IDENTIFIER, OUTPUT_DIR)
	print(f"running make_output_filename: {output}")
