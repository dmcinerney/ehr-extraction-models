import numpy as np
import torch
from pytt.batching.postprocessor import StandardOutputBatch
from .model import statistics_func


class OutputBatchApplications(StandardOutputBatch):
    @classmethod
    def from_outputs(cls, postprocessor, batch, outputs):
        loss, stats = cls.get_loss_stats(batch, outputs)
        with torch.autograd.no_grad():
            return loss, cls(len(batch), stats, batch=batch, outputs=outputs)

    @classmethod
    def loss(cls, batch, outputs):
        return None

    @classmethod
    def stats(cls, batch, outputs):
        results, stats = cls.test_func(batch, **outputs)
        stats['results'] = results
        return stats

    @classmethod
    def test_func(cls, batch, total_num_codes, num_codes, attention, traceback_attention, article_sentences_lengths, clustering, codes=None, labels=None):
        results = {'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths,
                   'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports'],
                   'clustering':clustering}
        if codes is not None:
            stats = statistics_func(total_num_codes, num_codes, attention, traceback_attention, article_sentences_lengths, clustering, codes, labels=labels)
        else:
            stats = {}
        return results, stats

    def __init__(self, batch_length, batch_stats, batch=None, outputs=None):
        super(OutputBatchApplications, self).__init__(batch_length, batch_stats, batch=batch, outputs=outputs)
        if 'results' in batch_stats.keys():
            self.results = batch_stats['results']
            del batch_stats['results']

class OutputBatchTest(OutputBatchApplications):
    @classmethod
    def from_outputs(cls, postprocessor, batch, outputs):
        postprocessor.add_summary_stats(batch, outputs)
        return super(OutputBatchTest, cls).from_outputs(postprocessor, batch, outputs)
