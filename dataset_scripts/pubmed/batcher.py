import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance,\
                                           StandardBatch
from pytt.utils import pad_and_concat

class PubmedBatcher(StandardBatcher):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def process_datapoint(self, raw_datapoint):
        return PubmedInstance(raw_datapoint, self.tokenizer)

class PubmedInstance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer):
        self.datapoint = raw_datapoint
        self.tensors = {}
        tokens = [tokenizer.tokenize(sent) for sent in raw_datapoint['article_sentences']]
        self.tensors['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokens])
        self.tensors['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokens])
        self.observed_keys = ['article_sentences', 'article_sentences_lengths']
        self.target_keys = []
