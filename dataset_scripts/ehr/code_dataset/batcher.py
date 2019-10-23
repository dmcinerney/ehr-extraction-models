import torch
from transformers import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance
from pytt.utils import pad_and_concat
import  models.clinical_bert.parameters as p
import spacy
nlp = spacy.load('en_core_web_sm')

class Batcher(StandardBatcher):
    def __init__(self, code_graph, simplify=True):
        self.code_graph = code_graph
        self.code_idxs = {code:i for i,code in enumerate(sorted(code_graph.nodes))}
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)
        self.simplify = True

    def process_datapoint(self, raw_datapoint):
        return Instance(raw_datapoint, self.tokenizer, self.code_idxs, self.simplify)

class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes, simplify):
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
        self.datapoint['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized_sentences])
        self.datapoint['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokenized_sentences])
        targets = raw_datapoint['targets']
        if simplify:
            targets = [target for target in targets]
        self.datapoint['codes'] = torch.tensor([codes[code_str] for code_str in targets])
        self.datapoint['num_codes'] = torch.tensor(self.datapoint['codes'].size(0))
        self.observed = ['article_sentences','article_sentences_lengths', 'codes', 'num_codes']
        if 'labels' in raw_datapoint.keys():
            self.datapoint['labels'] = torch.tensor(raw_datapoint['labels'])
        self.tokenized_sentences = tokenized_sentences

    def keep_in_batch(self):
        return {'tokenized_sentences':self.tokenized_sentences}
