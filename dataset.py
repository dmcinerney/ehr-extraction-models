import pandas as pd
from pytt.preprocessing.raw_dataset import RawDataset

class Dataset(RawDataset):
    def __init__(self, df):
        self.df = df

def init_dataset(filename):
    df = pd.read_json(filename, lines=True, compression='gzip')
    return Dataset(df)

def split_dataset(filename, split=.9):
    df = pd.read_json(filename, lines=True, compression='gzip').sample(frac=1)
    n = int(round(split * len(df)))
    return Dataset(df.iloc[:n]), Dataset(df.iloc[n:])
