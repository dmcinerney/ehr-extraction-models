import torch
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance,\
                                           StandardBatch
from pytt.utils import pad_and_concat


class Batcher(StandardBatcher):
    def process_datapoint(self, raw_datapoint):
        return Instance(raw_datapoint)

class SentenceBatcher(Batcher):
    def process_datapoint(self, raw_datapoint):
        return SentenceInstance(raw_datapoint)

class EvidenceBatcher(Batcher):
    def process_datapoint(self, raw_datapoint):
        return EvidenceInstance(raw_datapoint)

class Instance(StandardInstance):
    def __init__(self, raw_datapoint):
        self.datapoint = raw_datapoint
        self.tensors = {}
        self.tensors['article'] = torch.cat(
            [torch.tensor(sent) for sent in raw_datapoint['article'] if len(sent) > 0], 0)
        self.tensors['I'] = torch.tensor(raw_datapoint['I'])
        self.tensors['C'] = torch.tensor(raw_datapoint['C'])
        self.tensors['O'] = torch.tensor(raw_datapoint['O'])
        self.tensors['article_length'] = torch.tensor(self.tensors['article'].shape[0])
        self.tensors['I_length'] = torch.tensor(len(raw_datapoint['I']))
        self.tensors['C_length'] = torch.tensor(len(raw_datapoint['C']))
        self.tensors['O_length'] = torch.tensor(len(raw_datapoint['O']))
        self.tensors['evidence'] = torch.cat(
            [torch.tensor(evidence) for evidence in raw_datapoint['evidence'] if len(evidence) > 0], 0) == 1
        self.tensors['y'] = torch.tensor(raw_datapoint['y'])
        self.observed_keys = ['article','I','C','O','article_length','I_length','C_length','O_length','y']
        self.target_keys = ['evidence']

class SentenceInstance(Instance):
    def __init__(self, raw_datapoint):
        super(SentenceInstance, self).__init__(raw_datapoint)
        del self.tensors['article']
        del self.tensors['article_length']
        self.tensors['article_sentences'] = pad_and_concat(
            [torch.tensor(sent) for sent in raw_datapoint['article'] if len(sent) > 0])
        self.tensors['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in raw_datapoint['article'] if len(sent) > 0])
        self.tensors['evidence'] = pad_and_concat(
            [torch.tensor(evidence) for evidence in raw_datapoint['evidence'] if len(evidence) > 0])
        self.observed_keys = ['article_sentences','I','C','O','article_sentences_lengths','I_length','C_length','O_length', 'y']

class EvidenceInstance(SentenceInstance):
    def __init__(self, raw_datapoint):
        super(EvidenceInstance, self).__init__(raw_datapoint)
        self.tensors['article_sentences'] = self.tensors['article_sentences'][self.tensors['evidence'] == 1].unsqueeze(0)
        self.tensors['article_sentences_lengths'] = torch.tensor([self.tensors['article_sentences'].size(1)])
