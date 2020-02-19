import numpy as np
import torch
from pytt.utils import pad_and_concat
from ..code_supervision.postprocessor import OutputBatch as OB
from .model import statistics_func, loss_func
from utils import precision_recall_f1

class OutputBatch(OB):
    @classmethod
    def loss(cls, batch, outputs):
        return loss_func(**outputs)

    @classmethod
    def stats(cls, batch, outputs):
        returned_stats = statistics_func(**outputs)
        counts = pad_and_concat([returned_stats['true_positives'],
                                 returned_stats['positives'],
                                 returned_stats['relevants']],
                                auto=False)
        del returned_stats['true_positives']
        del returned_stats['positives']
        del returned_stats['relevants']
        returned_stats = {k:v.item() for k,v in returned_stats.items()}
        returned_stats['counts'] = counts
        return returned_stats

class OutputBatchTest(OutputBatch):
    @classmethod
    def from_outputs(cls, postprocessor, batch, outputs):
        postprocessor.add_summary_stats(batch, outputs)
        return super(OutputBatchTest, cls).from_outputs(postprocessor, batch, outputs)

    def __init__(self, *args, **kwargs):
        kwargs['outputs'] = None
        super(OutputBatchTest, self).__init__(*args, **kwargs)

    def write_to_tensorboard(self, *args, **kwargs):
        return
        super(OutputBatchTest, self).write_to_tensorboard(*args, **kwargs)
        self.outputs = {
            'scores':self.outputs['scores'][:0],
            'labels':self.outputs['labels'][:0],
            'num_codes':self.outputs['num_codes'][:0]}


class OutputBatchApplications(OutputBatch):
    @classmethod
    def loss(cls, batch, outputs):
        if 'labels' in outputs.keys():
            return loss_func(**outputs)
        else:
            return None

    @classmethod
    def stats(cls, batch, outputs):
        results, stats = cls.test_func(batch, **outputs)
        stats['results'] = results
        return stats

    @classmethod
    def test_func(cls, batch, total_num_codes, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels=None):
        # TODO: make result from batch
        sentence_level_attentions = get_sentence_level_attentions(sentence_level_scores, article_sentences_lengths, torch.ones_like(scores).byte())
        attention = get_full_attention(word_level_attentions, sentence_level_attentions)
        traceback_attention = get_full_attention(traceback_word_level_attentions, sentence_level_attentions)
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths,
                   'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        if labels is not None:
            stats = statistics_func(total_num_codes, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels)
        else:
            stats = {}
        return results, stats

    def __init__(self, batch_length, batch_stats, **kwargs):
        kwargs['outputs'] = None
        super(OutputBatchApplications, self).__init__(batch_length, batch_stats, **kwargs)
        if 'results' in batch_stats.keys():
            self.results = batch_stats['results']
            del batch_stats['results']
