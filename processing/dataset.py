import pandas as pd
from pytt.preprocessing.raw_dataset import RawDataset

class Dataset(RawDataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, i):
        dictionary = super(Dataset, self).__getitem__(i)
        dictionary['reports'] = pd.DataFrame(eval(dictionary['reports']))
        dictionary['reports']['date'] = pd.to_datetime(dictionary['reports']['date'])
        dictionary['targets'] = eval(dictionary['targets'])
        dictionary['labels'] = eval(dictionary['labels'])
        return dictionary

def init_dataset(filename, limit_rows=None):
    if limit_rows is None:
        df = pd.read_csv(filename, compression='gzip')
    else:
        df = next(iter(pd.read_csv(filename, compression='gzip', chunksize=limit_rows)))
    return Dataset(df)

def split_dataset(filename, split=.9):
    df = pd.read_json(filename, lines=True, compression='gzip').sample(frac=1)
    n = int(round(split * len(df)))
    return Dataset(df.iloc[:n]), Dataset(df.iloc[n:])
