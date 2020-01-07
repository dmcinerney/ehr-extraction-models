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
    def __init__(self, code_graph, ancestors=False, code_id=False, code_description=False, code_linearization=False):
        self.code_graph = code_graph
        self.code_idxs = {code:i for i,code in enumerate(sorted(code_graph.nodes))}
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)
        self.ancestors = ancestors
        self.code_id = code_id
        self.code_description = code_description
        self.code_linearization = code_linearization

    def process_datapoint(self, raw_datapoint):
        return Instance(raw_datapoint,
                        self.tokenizer,
                        self.code_idxs,
                        self.code_graph,
                        ancestors=self.ancestors,
                        code_id=self.code_id,
                        code_description=self.code_description,
                        code_linearization=self.code_linearization)


def process_reports(tokenizer, reports_df):
    reports = [nlp(report.text) for i,report in reports_df.iterrows()]
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
    return tokenized_sentences, sentence_spans

class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes, code_graph, ancestors=False, code_id=False, code_description=False, code_linearization=False):
        self.raw_datapoint = raw_datapoint
        self.datapoint = {}
        self.observed = []

        # process_reports
        k = 100
        raw_datapoint['reports'] = raw_datapoint['reports'][-k:]
        self.tokenized_sentences, self.sentence_spans = process_reports(tokenizer, raw_datapoint['reports'])
        self.datapoint['article_sentences'] = pad_and_concat(
            [torch.tensor(tokenizer.convert_tokens_to_ids(sent)) for sent in self.tokenized_sentences])
        self.datapoint['article_sentences_lengths'] = torch.tensor(
            [len(sent) for sent in self.tokenized_sentences])
        self.observed += ['article_sentences', 'article_sentences_lengths']

        # get code_id
        # this happens regardless of whether it is observed because it might be needed for supervision
        if 'targets' in raw_datapoint.keys():
            # only gets codes it when given targets
            targets = raw_datapoint['targets']
            self.datapoint['codes'] = torch.tensor([codes[code_str] for code_str in targets])
            self.datapoint['num_codes'] = torch.tensor(self.datapoint['codes'].size(0))
            self.observed += ['num_codes']

        # get observed
        if code_id:
            # needs targets
            if 'targets' not in raw_datapoint.keys():
                raise Exception
            self.observed += ['codes']
        if code_description:
            # get description
            # doesn't need targets as long as it has queries
            if 'targets' in raw_datapoint.keys():
                descriptions = [code_graph.nodes[code_str]['description'] if 'description' in code_graph.nodes[code_str].keys() else 'none' for code_str in raw_datapoint['targets']]
            else:
                descriptions = raw_datapoint['queries']
                # if targets were not given, you still need num_codes
                self.datapoint['num_codes'] = torch.tensor(len(descriptions))
                self.observed += ['num_codes']
            tokenized_descriptions = [[tokenizer.cls_token] + tokenizer.tokenize(d) + [tokenizer.sep_token] for d in descriptions]
            self.datapoint['code_description'] = pad_and_concat(
                [torch.tensor(tokenizer.convert_tokens_to_ids(d)) for d in tokenized_descriptions])
            self.datapoint['code_description_length'] = torch.tensor(
                [len(d) for d in tokenized_descriptions])
            self.observed += ['code_description', 'code_description_length']
        if code_linearization:
            # get code linearization
            # needs targets
            if 'targets' not in raw_datapoint.keys():
                raise Exception
            raise NotImplementedError

        if 'labels' in raw_datapoint.keys():
            self.datapoint['labels'] = torch.tensor(raw_datapoint['labels'])

    def keep_in_batch(self):
        return {'tokenized_sentences':self.tokenized_sentences, 'sentence_spans':self.sentence_spans, 'original_reports':self.raw_datapoint['reports']}
