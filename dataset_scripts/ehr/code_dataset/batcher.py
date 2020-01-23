import random
import copy
import torch
from transformers import BertTokenizer
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance,\
                                           StandardBatch
from pytt.utils import pad_and_concat
import models.clinical_bert.parameters as p
import spacy
import pandas as pd
nlp = spacy.load('en_core_web_sm')

class Batcher(StandardBatcher):
    # Note: ancestors is done in preprocessing sometimes
    def __init__(self, code_graph, ancestors=False, code_id=False, code_description=False, code_linearization=False, sample_top=None):
        self.graph_ops = GraphOps(code_graph)
        self.code_idxs = {code:i for i,code in enumerate(sorted(code_graph.nodes))}
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)
        self.ancestors = ancestors
        self.code_id = code_id
        self.code_description = code_description
        self.code_linearization = code_linearization
        self.sample_top = sample_top

    def process_datapoint(self, raw_datapoint):
        return Instance(raw_datapoint,
                        self.tokenizer,
                        self.code_idxs,
                        self.graph_ops,
                        ancestors=self.ancestors,
                        code_id=self.code_id,
                        code_description=self.code_description,
                        code_linearization=self.code_linearization,
                        sample_top=self.sample_top)

    def batch(self, instances, devices=None):
        return super(Batcher, self).batch(instances, devices=devices, batch_class=create_batch_class(len(self.code_idxs)))

def process_reports(tokenizer, reports_df, num_sentences=None):
    reports_iter = list(reports_df.iterrows())
    if num_sentences is not None and num_sentences < 0:
        reports_iter = reversed(reports_iter)
        append_func = lambda x, y: x.insert(0, y)
    else:
        append_func = lambda x, y: x.append(y)
    tokenized_sentences = []
    sentence_spans = []
    sentence_count = 0
    for i,report in reports_iter:
        report_sentence_spans = []
        sents = list(nlp(report.text).sents)
        if num_sentences is not None and num_sentences < 0:
            sents = reversed(sents)
        for sent in sents:
            if num_sentences is not None and sentence_count >= abs(num_sentences):
                break
            tokenized_sent = tokenizer.tokenize(sent.text)
            if len(tokenized_sent) > 4: # NOTE THIS IS HARDCODED
                append_func(tokenized_sentences, [tokenizer.cls_token] + tokenized_sent + [tokenizer.sep_token])
                append_func(report_sentence_spans, (sent.start_char, sent.end_char))
                sentence_count += 1
        append_func(sentence_spans, report_sentence_spans)
    j = 0
    for report_sentence_spans in sentence_spans:
        for i,sentence_span in enumerate(report_sentence_spans):
            report_sentence_spans[i] = (j, *sentence_span)
            j += 1
    return tokenized_sentences, sentence_spans

def get_pos_neg(targets, labels):
    negatives = []
    positives = []
    for i,target in enumerate(targets):
        if labels[i]:
            positives.append(target)
        else:
            negatives.append(target)
    return positives, negatives

def get_targets_labels(pos, neg):
    return pos + neg, [1]*len(pos) + [0]*len(neg)

class GraphOps:
    # ASSUMES THE GRAPH IS A DAG WITH ONLY ONE NODE OF IN_DEGREE 0!
    def __init__(self, graph):
        self.graph = graph
        self.node_option_idx = {}
        self.node_idx_option = {}
        self.max_index = -1
        for n in self.graph.nodes:
            if self.graph.in_degree(n) == 0:
                self.start_node = n
            self.node_idx_option[n] = []
            for i,succ in enumerate(self.graph.successors(n)):
                self.node_option_idx[succ] = i
                self.node_idx_option[n].append(succ)
                self.max_index = max(self.max_index, i)

    def ancestors(self, nodes, stop_nodes=set()):
        node_stack = copy.deepcopy(nodes)
        new_nodes = set()
        while len(node_stack) > 0:
            node = node_stack.pop()
            if node in stop_nodes: continue # don't add stop nodes
            if node in new_nodes: continue # don't add nodes already there
            in_degree = self.graph.in_degree(node)
            if in_degree == 0: continue # don't add the start node
            elif in_degree > 1: raise Exception # shouldn't have any nodes with more than one parent
            new_nodes.add(node)
            node_stack.extend(list(graph.predecessors(node)))
        return list(new_nodes)

    def get_descriptions(self, nodes):
        return [self.graph.nodes[node]['description'] for node in nodes]

    def linearize(self, node):
        backwards_options = []
        while self.graph.in_degree(node) > 0:
            backwards_options.append(self.node_option_idx[node])
            node = next(iter(self.graph.predecessors(node)))
        return list(reversed(backwards_options))

    def delinearize(self, linearized_node):
        node = self.start_node
        for i in linearized_node:
            node = self.node_idx_option[node][i]
        return node

class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes, graph_ops, ancestors=False, code_id=False, code_description=False, code_linearization=False, sample_top=None):
        self.raw_datapoint = raw_datapoint
        self.datapoint = {}
        self.observed = []

        # process_reports
        num_sentences = -1000
        if 'num_sentences' in raw_datapoint.keys():
            num_sentences = raw_datapoint['num_sentences']
        self.tokenized_sentences, self.sentence_spans = process_reports(tokenizer, raw_datapoint['reports'], num_sentences=num_sentences)
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
            if 'labels' in raw_datapoint.keys():
                labels = raw_datapoint['labels']
                if ancestors or sample_top is not None:
                    positives, negatives = get_pos_neg(targets, labels)
                if ancestors:
                    positives = graph_ops.ancestors(positives)
                    negatives = graph_ops.ancestors(negatives, stop_nodes=positives)
                if sample_top is not None:
                    num_pos_samples = min(len(positives), sample_top/2)
                    positives = random.sample(positives, num_pos_samples)
                    num_neg_samples = min(len(negatives), sample_top - num_pos_samples)
                    negatives = random.sample(negatives, num_neg_samples)
                if ancestors or sample_top is not None:
                    targets, labels = get_targets_labels(positives, negatives)
                self.datapoint['labels'] = torch.tensor(labels)
            self.datapoint['codes'] = torch.tensor([codes[code_str] for code_str in targets])
            self.datapoint['num_codes'] = torch.tensor(self.datapoint['codes'].size(0))
            self.observed += ['num_codes']
        elif 'labels' in raw_datapoint.keys():
            raise Exception

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
                descriptions = graph_ops.get_descriptions(targets)
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
            linearized_codes = [graph_ops.linearize(target) for target in targets]
            self.datapoint['linearized_codes'] = pad_and_concat(
                [torch.tensor(linearized_code) for linearized_code in linearized_codes])
            self.datapoint['linearized_codes_lengths'] = torch.tensor(
                [len(linearized_code) for linearized_code in linearized_codes])
            self.observed += ['linearized_codes', 'linearized_codes_lengths']

    def keep_in_batch(self):
        return {'tokenized_sentences':self.tokenized_sentences, 'sentence_spans':self.sentence_spans, 'original_reports':self.raw_datapoint['reports']}

def create_batch_class(total_num_codes):
    class Batch(StandardBatch):
        def get_target(self):
            return dict(**super(Batch, self).get_target(), total_num_codes=torch.tensor(total_num_codes))
    return Batch
