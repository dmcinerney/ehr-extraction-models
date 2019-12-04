import torch
from transformers import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance
from pytt.utils import pad_and_concat
import models.clinical_bert.parameters as p
import spacy
import pandas as pd
nlp = spacy.load('en_core_web_sm')

class Batcher(StandardBatcher):
    def __init__(self, code_graph, instance_type=None):
        self.code_graph = code_graph
        self.code_idxs = {code:i for i,code in enumerate(sorted(code_graph.nodes))}
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)
        self.instance_type = instance_type

    def process_datapoint(self, raw_datapoint):
        if self.instance_type is None:
            return Instance(raw_datapoint, self.tokenizer, self.code_idxs, self.code_graph)
        elif self.instance_type == 'with_description':
            return InstanceWithDescription(raw_datapoint, self.tokenizer, self.code_idxs, self.code_graph)
        elif self.instance_type == 'only_description':
            return InstanceOnlyDescription(raw_datapoint, self.tokenizer, self.code_idxs, self.code_graph)

class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes, code_graph):
        self.raw_datapoint = raw_datapoint
        self.datapoint = {}
        k = 100
        raw_datapoint['reports'] = raw_datapoint['reports'][-k:]
        reports = [nlp(report.text) for i,report in raw_datapoint['reports'].iterrows()]
        tokenized_sentences = []
        sentence_spans = []
        j = 0
        for i,report in enumerate(reports):
            report_sentence_spans = []
            for sent in report.sents:
                tokenized_sent = tokenizer.tokenize(sent.text)
                if len(tokenized_sent) < 4: continue
                if len(tokenized_sent) > 0:
                    tokenized_sentences.append([tokenizer.cls_token] + tokenized_sent + [tokenizer.sep_token])
                    report_sentence_spans.append((j, sent.start_char, sent.end_char))
                    j += 1
            sentence_spans.append(report_sentence_spans)
        self.datapoint['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized_sentences])
        self.datapoint['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in tokenized_sentences])
        if 'targets' in raw_datapoint.keys():
            targets = raw_datapoint['targets']
            self.datapoint['codes'] = torch.tensor([codes[code_str] for code_str in targets])
        else:
            targets = raw_datapoint['queries']
            self.datapoint['codes'] = torch.tensor([-1]*len(targets))
        self.datapoint['num_codes'] = torch.tensor(self.datapoint['codes'].size(0))
        self.observed = ['article_sentences','article_sentences_lengths', 'codes', 'num_codes']
        if 'labels' in raw_datapoint.keys():
            self.datapoint['labels'] = torch.tensor(raw_datapoint['labels'])
        self.tokenized_sentences = tokenized_sentences
        self.sentence_spans = sentence_spans

    def keep_in_batch(self):
        return {'tokenized_sentences':self.tokenized_sentences, 'sentence_spans':self.sentence_spans, 'original_reports':self.raw_datapoint['reports']}

class InstanceWithDescription(Instance):
    def __init__(self, raw_datapoint, tokenizer, codes, code_graph):
        super(InstanceWithDescription, self).__init__(raw_datapoint, tokenizer, codes, code_graph)
        if 'targets' in raw_datapoint.keys():
            descriptions = [code_graph.nodes[code_str]['description'] if 'description' in code_graph.nodes[code_str].keys() else 'none' for code_str in raw_datapoint['targets']]
        else:
            descriptions = raw_datapoint['queries']
        tokenized_descriptions = [[tokenizer.cls_token] + tokenizer.tokenize(d) + [tokenizer.sep_token] for d in descriptions]
        self.datapoint['code_description'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(d)) for d in tokenized_descriptions])
        self.datapoint['code_description_length'] = torch.tensor(
            [len(d) for d in tokenized_descriptions])
        self.observed += ['code_description', 'code_description_length']

class InstanceOnlyDescription(InstanceWithDescription):
    def __init__(self, raw_datapoint, tokenizer, codes, code_graph):
        super(InstanceOnlyDescription, self).__init__(raw_datapoint, tokenizer, codes, code_graph)
        del self.observed[2]
