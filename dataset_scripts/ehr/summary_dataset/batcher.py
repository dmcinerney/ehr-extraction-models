import torch
from transformers import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance
from pytt.utils import pad_and_concat
import  models.clinical_bert.parameters as p
import spacy
nlp = spacy.load('en_core_web_sm')

class Batcher(StandardBatcher):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)

class SupervisedBatcher(Batcher):
    def process_datapoint(self, raw_datapoint):
        return SupervisedInstance(raw_datapoint, self.tokenizer)

class UnsupervisedBatcher(Batcher):
    def process_datapoint(self, raw_datapoint):
        return UnsupervisedInstance(raw_datapoint, self.tokenizer)

class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer):
        self.datapoint = raw_datapoint
        self.tensors = {}
        max_length = 1000000
        text = raw_datapoint['reports']
        text = [nlp(text[offset:offset+max_length]) for offset in range(0,len(text),max_length)]
        tokenized_sentences = []
        for section in text:
            for sent in section.sents:
                tokenized_sent = tokenizer.tokenize(sent.text)
                if len(tokenized_sent) > 0:
                    tokenized_sentences.append(tokenized_sent)
        self.tensors['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized_sentences])
        self.tensors['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokenized_sentences])
        import pdb; pdb.set_trace()
        # TODO: add summary stuff in
        self.datapoint['tokenized_sentences'] = tokenized_sentences
        self.set_observed_and_target_keys()

    def set_observed_and_target_keys():
        raise NotImplementedError

class SupervisedInstance(Instance):
    def set_observed_and_target_keys():
        self.observed_keys = ['article_sentences', 'article_sentences_lengths', 'summary', 'summary_length']
        self.target_keys = []

class UnsupervisedInstance(Instance):
    def set_observed_and_target_keys():
        self.observed_keys = ['article_sentences', 'article_sentences_lengths']
        self.target_keys = ['summary', 'summary_length']
