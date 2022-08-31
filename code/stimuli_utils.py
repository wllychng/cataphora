
# this file contains various functions for cleaning/processing the stimuli data

import pandas as pd

def clean_stimuli(df):
    # delete rows where stimuli is NaN
    assert "example" in df.columns
    df.dropna(subset = ["example"], inplace = True)
    return df

# make sure dataset loaded into df conforms to expectations of stimuli data,
# throw errors otherwise
def validate_df(df):
	print("TODO")
	return True

if __name__ == "__main__":
    # test clean_stimuli function
    test_df = pd.DataFrame({"col1": [1, 2, 3], "example": ["A", None, "C"]})
    test_df = clean_stimuli(test_df)
    print(test_df)
