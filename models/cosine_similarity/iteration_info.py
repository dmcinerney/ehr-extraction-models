import numpy as np
import torch
from pytt.iteration_info import BatchInfo as BI
from .model import statistics_func


class BatchInfo(BI):
    def stats(self):
        return statistics_func(**self.batch_outputs)

class BatchInfoApplications(BatchInfo):
    def stats(self):
        self.results, stats = self.test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

    def test_func(self, batch, total_num_codes, code_idxs, num_codes, attention, traceback_attention, article_sentences_lengths, codes=None, labels=None):
        results = {'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths,
                   'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        if codes is not None:
            stats = statistics_func(total_num_codes, code_idxs, num_codes, attention, traceback_attention, article_sentences_lengths, codes, labels=labels)
        else:
            stats = {}
        return results, stats
