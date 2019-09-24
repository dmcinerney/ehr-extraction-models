import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance
from pytt.utils import pad_and_concat
import spacy
nlp = spacy.load('en_core_web_sm')

class EHRBatcher(StandardBatcher):
    def __init__(self, codes):
        self.codes = codes
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def process_datapoint(self, raw_datapoint):
        return EHRInstance(raw_datapoint, self.tokenizer, self.codes)

class EHRInstance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes):
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
        self.tensors['codes'] = torch.tensor([codes[code_str] for code_str in raw_datapoint['targets']])
        self.tensors['num_codes'] = torch.tensor(self.tensors['codes'].size(0))
        self.observed_keys = ['article_sentences','article_sentences_lengths', 'codes', 'num_codes']
        if 'labels' in raw_datapoint.keys():
            self.tensors['labels'] = torch.tensor(raw_datapoint['labels'])
            self.target_keys = ['labels']
        else:
            self.target_keys = []
        self.datapoint['tokenized_sentences'] = tokenized_sentences
