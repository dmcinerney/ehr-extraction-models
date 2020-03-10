import numpy as np
import torch
from pytt.batching.standard_batcher import StandardBatcher,\
                                           StandardInstance,\
                                           StandardBatch
from pytt.utils import pad_and_concat
import spacy
import pandas as pd
from .tokenizer import TfidfTokenizerWrapper, BertTokenizerWrapper
from hierarchy import Hierarchy
nlp = spacy.load('en_core_web_sm')


class Batcher(StandardBatcher):
    # NOTE: ancestors is done in preprocessing sometimes
    def __init__(self, hierarchy, ancestors=False, code_id=False, code_description=False, code_linearization=False, description_linearization=False, description_embedding_linearization=False, resample_neg_proportion=None, counts=None, tfidf_tokenizer=False, add_special_tokens=True):
        self.hierarchy = hierarchy
        # TODO: delete first line when no more old models need to be trained further, newer models use second line
        #self.code_idxs = {code:i for i,code in enumerate(sorted(hierarchy.get_nodes()))}
        self.code_idxs = {code:i for i,code in enumerate(sorted(hierarchy.descriptions.keys()))}
        self.tfidf_tokenizer = tfidf_tokenizer
        if tfidf_tokenizer:
            self.tokenizer = TfidfTokenizerWrapper()
        else:
            self.tokenizer = BertTokenizerWrapper(with_cls=add_special_tokens, with_sep=add_special_tokens)
        bert_tokenizer = BertTokenizerWrapper() # need a special bert_tokenizer with added special tokens to filter sentences
        self.filter = lambda x: len(bert_tokenizer.tokenize(x)) > 4 # NOTE: this is hardcoded

        self.ancestors = ancestors
        self.code_id = code_id
        self.code_description = code_description
        self.code_linearization = code_linearization
        self.description_linearization = description_linearization
        self.description_embedding_linearization = description_embedding_linearization
        self.resample_neg_proportion = resample_neg_proportion
        if resample_neg_proportion is not None:
            if counts is None:
                raise Exception
            keys, values = zip(*counts.items())
            self.counts = pd.DataFrame(values, index=keys, columns=['negative', 'positive'])
        else:
            self.counts = None

    def get_code_embedding_types(self):
        code_embedding_types = set([])
        if self.code_id:
            code_embedding_types.add('codes')
        if self.code_description:
            code_embedding_types.add('descriptions')
        if self.code_linearization:
            code_embedding_types.add('linearized_codes')
        if self.description_linearization:
            code_embedding_types.add('linearized_descriptions')
        if self.description_embedding_linearization:
            code_embedding_types.add('linearized_description_embeddings')
        return code_embedding_types

    def process_datapoint(self, raw_datapoint):
        return Instance(raw_datapoint,
                        self.tokenizer,
                        self.code_idxs,
                        self.hierarchy,
                        ancestors=self.ancestors,
                        code_id=self.code_id,
                        code_description=self.code_description,
                        code_linearization=self.code_linearization,
                        description_linearization=self.description_linearization,
                        description_embedding_linearization=self.description_embedding_linearization,
                        resample_neg_proportion=self.resample_neg_proportion,
                        counts=self.counts,
                        tfidf_tokenizer=self.tfidf_tokenizer,
                        filter=self.filter)

def get_sentences(reports_df, num_sentences=None, filter=lambda x: True, report_max_length=1000000):
    reports_iter = list(reports_df.iterrows())
    if num_sentences is not None and num_sentences < 0:
        reports_iter = reversed(reports_iter)
        append_func = lambda x, y: x.insert(0, y)
    else:
        append_func = lambda x, y: x.append(y)
    sentences = []
    sentence_spans = []
    sentence_count = 0
    for i,report in reports_iter:
        report_sentence_spans = []
        sents = list(nlp(report.text[:report_max_length]).sents)
        if num_sentences is not None and num_sentences < 0:
            sents = reversed(sents)
        for sent in sents:
            if num_sentences is not None and sentence_count >= abs(num_sentences):
                break
            if filter(sent.text):
                append_func(sentences, sent.text)
                append_func(report_sentence_spans, (sent.start_char, sent.end_char))
                sentence_count += 1
        append_func(sentence_spans, report_sentence_spans)
    j = 0
    for report_sentence_spans in sentence_spans:
        for i,sentence_span in enumerate(report_sentence_spans):
            report_sentence_spans[i] = (j, *sentence_span)
            j += 1
    return sentences, sentence_spans


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


class Instance(StandardInstance):
    def __init__(self, raw_datapoint, tokenizer, codes, hierarchy, ancestors=False, code_id=False, code_description=False, code_linearization=False, description_linearization=False, description_embedding_linearization=False, resample_neg_proportion=None, counts=None, tfidf_tokenizer=False, filter=lambda x: True):
        self.raw_datapoint = raw_datapoint
        self.datapoint = {}
        self.observed = []

        # process_reports
        num_sentences = -1000
        if 'num_sentences' in raw_datapoint.keys():
            num_sentences = raw_datapoint['num_sentences']
        sentences, self.sentence_spans = get_sentences(raw_datapoint['reports'], num_sentences=num_sentences, filter=filter)
        self.tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
        if not tfidf_tokenizer:
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
                if ancestors or resample_neg_proportion is not None:
                    positives, negatives = get_pos_neg(targets, labels)
                if ancestors:
                    positives = hierarchy.ancestors(positives)
                    negatives = hierarchy.ancestors(negatives, stop_nodes=positives)
                if resample_neg_proportion is not None:
                    # sample negative according to positive prior for that code
                    total_negatives = counts.negative.sum()
                    individual_probs = np.array([(counts.positive[c]/(counts.positive.sum()))
                                                 * (1/counts.negative[c]) for c in negatives])
                    individual_probs = individual_probs/individual_probs.sum()
                    negatives = list(np.random.choice(negatives, size=len(negatives), p=individual_probs))
                    keep = np.random.binomial(len(negatives), resample_neg_proportion)
                    negatives = negatives[:keep]
                if ancestors or resample_neg_proportion is not None:
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
                descriptions = (d if d is not None else t for t,d in zip(targets,hierarchy.get_descriptions(targets)))
            else:
                descriptions = raw_datapoint['queries']
                # if targets were not given, you still need num_codes
                self.datapoint['num_codes'] = torch.tensor(len(descriptions))
                self.observed += ['num_codes']
            tokenized_descriptions = [tokenizer.tokenize(d) for d in descriptions]
            if not tfidf_tokenizer:
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
            linearized_codes = [hierarchy.linearize(target) for target in targets]
            self.datapoint['linearized_codes'] = pad_and_concat(
                [torch.tensor(linearized_code) for linearized_code in linearized_codes])
            self.datapoint['linearized_codes_lengths'] = torch.tensor(
                [len(linearized_code) for linearized_code in linearized_codes])
            self.observed += ['linearized_codes', 'linearized_codes_lengths']
        if description_linearization:
            # get description
            # doesn't need targets as long as it has queries
            if 'targets' in raw_datapoint.keys():
                descriptions = [get_description_linearization(t, hierarchy) for t in targets]
            else:
                raise NotImplementedError # interface doesn't produce valid queries for this yet
                descriptions = raw_datapoint['queries']
                # if targets were not given, you still need num_codes
                self.datapoint['num_codes'] = torch.tensor(len(descriptions))
                self.observed += ['num_codes']
            tokenized_description_linearizations = [tokenizer.tokenize(d) for d in descriptions]
            if not tfidf_tokenizer:
                self.datapoint['linearized_descriptions'] = pad_and_concat(
                    [torch.tensor(tokenizer.convert_tokens_to_ids(d)) for d in tokenized_description_linearizations])
                self.datapoint['linearized_descriptions_lengths'] = torch.tensor(
                    [len(d) for d in tokenized_description_linearizations])
                self.observed += ['linearized_descriptions', 'linearized_descriptions_lengths']
        if description_embedding_linearization:
            # get description
            # doesn't need targets as long as it has queries
            if 'targets' in raw_datapoint.keys():
                descriptions = [get_description_embedding_linearization(t, hierarchy) for t in targets]
            else:
                raise NotImplementedError # interface doesn't produce valid queries for this yet
                descriptions = raw_datapoint['queries']
                # if targets were not given, you still need num_codes
                self.datapoint['num_codes'] = torch.tensor(len(descriptions))
                self.observed += ['num_codes']
            tokenized_description_embedding_linearizations = [tokenizer.tokenize(d) for d in descriptions]
            if not tfidf_tokenizer:
                self.datapoint['linearized_description_embeddings'] = pad_and_concat(
                    [torch.tensor(tokenizer.convert_tokens_to_ids(d)) for d in tokenized_description_embedding_linearizations])
                self.datapoint['linearized_description_embeddings_lengths'] = torch.tensor(
                    [len(d) for d in tokenized_description_embedding_linearizations])
                self.observed += ['linearized_description_embeddings', 'linearized_description_embeddings_lengths']

        if tfidf_tokenizer:
            if code_description:
                tfidf_matrix = tokenizer.convert_tokens_to_ids(self.tokenized_sentences+tokenized_descriptions)
                self.datapoint['code_description'] = torch.tensor(tfidf_matrix[len(self.tokenized_sentences):])
                self.observed += ['code_description']
                tfidf_matrix = tfidf_matrix[:len(self.tokenized_sentences)]
            else:
                tfidf_matrix = tokenizer.convert_tokens_to_ids(self.tokenized_sentences)
            self.datapoint['article_sentences'] = torch.tensor(tfidf_matrix)
            self.datapoint['num_sentences'] = torch.tensor(len(self.tokenized_sentences))
            self.observed += ['article_sentences', 'num_sentences']

    def keep_in_batch(self):
        keep_in_batch_dict = {'tokenized_sentences':self.tokenized_sentences, 'sentence_spans':self.sentence_spans, 'original_reports':self.raw_datapoint['reports']}
        if 'annotations' in self.raw_datapoint.keys():
            keep_in_batch_dict['annotations'] = self.raw_datapoint['annotations']
        return keep_in_batch_dict

def get_description_linearization(target, hierarchy):
    targets = hierarchy.path(target)
    return ' . '.join(d if d is not None else t for t,d in zip(targets,hierarchy.get_descriptions(targets)))
