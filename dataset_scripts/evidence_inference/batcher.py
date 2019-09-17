import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance,\
                                           StandardBatch
from pytt.utils import pad_and_concat


class EvidenceInferenceBatcher(StandardBatcher):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def process_datapoint(self, raw_datapoint):
        return EvidenceInferenceInstance(raw_datapoint, self.tokenizer)

class EvidenceInferenceInstance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer):
        self.datapoint = raw_datapoint
        tokens = []
        evidence = []
        for sentnum,sent in enumerate(raw_datapoint['article']):
            if len(sent) > 0:
                tokens.append([])
                evidence.append([])
                for tokennum,token in enumerate(sent):
                    for subtoken in tokenizer.tokenize(token):
                        tokens[-1].append(subtoken)
                        evidence[-1].append(raw_datapoint['evidence'][sentnum][tokennum])
        self.tensors = {}
        self.tensors['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokens])
        self.tensors['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokens])
        self.tensors['I'] = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(raw_datapoint['I']))))
        self.tensors['C'] = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(raw_datapoint['C']))))
        self.tensors['O'] = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(raw_datapoint['O']))))
        self.tensors['I_length'] = torch.tensor(len(raw_datapoint['I']))
        self.tensors['C_length'] = torch.tensor(len(raw_datapoint['C']))
        self.tensors['O_length'] = torch.tensor(len(raw_datapoint['O']))
        self.tensors['y'] = torch.tensor(raw_datapoint['y'])
        self.tensors['evidence'] = pad_and_concat(
            [torch.tensor(sent_ev) for sent_ev in evidence])
        self.observed_keys = ['article_sentences','article_sentences_lengths','I','C','O','I_length','C_length','O_length']
        self.target_keys = ['evidence', 'y']
