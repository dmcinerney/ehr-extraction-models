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
        self.raw_datapoint = raw_datapoint
        self.datapoint = {}
        max_length = 1000000
        text = raw_datapoint['reports']
        text = [nlp(text[offset:offset+max_length]) for offset in range(0,len(text),max_length)]
        tokenized_sentences = []
        for section in text:
            for sent in section.sents:
                tokenized_sent = tokenizer.tokenize(sent.text)
                if len(tokenized_sent) > 0:
                    tokenized_sentences.append([tokenizer.cls_token] + tokenized_sent + [tokenizer.sep_token])
        self.tokenized_sentences = tokenized_sentences
        self.datapoint['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized_sentences])
        self.datapoint['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokenized_sentences])
        text = raw_datapoint['impression']
#        text = 'hello my name is jered'
        self.tokenized_summary = [tokenizer.cls_token] + tokenizer.tokenize(text) + [tokenizer.sep_token]
        self.datapoint['summary'] = torch.tensor(tokenizer.convert_tokens_to_ids(self.tokenized_summary))
        self.datapoint['summary_length'] = torch.tensor(self.datapoint['summary'].size(0))
        self.set_observed()

    def set_observed(self):
        raise NotImplementedError

    def keep_in_batch(self):
        return {'tokenized_sentences':self.tokenized_sentences, 'tokenized_summary':self.tokenized_summary, 'df_index':self.raw_datapoint['df_index']}

class SupervisedInstance(Instance):
    def set_observed(self):
        self.observed = ['article_sentences', 'article_sentences_lengths', 'summary', 'summary_length']

class UnsupervisedInstance(Instance):
    def set_observed(self):
        self.observed = ['article_sentences', 'article_sentences_lengths']
