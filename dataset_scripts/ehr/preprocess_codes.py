import pandas as pd
import pickle as pkl

def get_codes(df, code_file=None):
    codes = set()
    for i,row in df.iterrows():
        for code in row.positive_targets:
            codes.add(code)
        for code in row.negative_targets:
            codes.add(code)
    if code_file is not None:
        with open(code_file, 'wb') as f:
            pkl.dump(list(codes), f)
    return codes

if __name__ == '__main__':
    filename = 'data/mimic/mimic_reports_to_codes.data'
    df = pd.read_json(filename, lines=True, compression='gzip')
    code_file = 'data/mimic/codes.pkl'
    get_codes(df, code_file=code_file)
