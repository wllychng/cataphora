
# this file contains various functions for cleaning/processing the stimuli data

import pandas as pd

def clean_stimuli(df):
    # delete rows where stimuli is NaN
    assert "example" in df.columns
    df.dropna(subset = "example", inplace = True)
    return df

if __name__ == "__main__":
    # test clean_stimuli function
    print("TODO")
