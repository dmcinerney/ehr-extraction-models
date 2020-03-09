import torch
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    params = torch.load(args.file)
    del params["code_embeddings.weight"]
    torch.save(params, args.file)
