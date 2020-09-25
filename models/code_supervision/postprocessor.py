import numpy as np
import torch
from pytt.utils import pad_and_concat
from pytt.batching.postprocessor import StandardOutputBatch
from .model import statistics_func, loss_func_creator
from utils import precision_recall_f1, plot_stacked_bar
import numpy as np
loss_func = loss_func_creator() # TODO: change this

class OutputBatch(StandardOutputBatch):
    @classmethod
    def from_outputs(cls, postprocessor, batch, outputs):
        loss, stats = cls.get_loss_stats(batch, outputs)
        with torch.autograd.no_grad():
            return loss, cls(len(batch), stats, batch=batch, outputs=outputs, postprocessor=postprocessor)

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

    def __init__(self, batch_length, batch_stats, batch=None, outputs=None, postprocessor=None):
        super(OutputBatch, self).__init__(batch_length, batch_stats, batch=batch, outputs=outputs)
        self.postprocessor = postprocessor
        if 'counts' in batch_stats.keys():
            self.counts = batch_stats['counts']
            del batch_stats['counts']
        if outputs is not None:
            self.outputs = {
                'scores':outputs['scores'],
                'labels':outputs['labels'],
                'num_codes':outputs['num_codes']}


    def __add__(self, output_batch):
        new_output_batch = super(OutputBatch, self).__add__(output_batch)
        new_output_batch.postprocessor = self.postprocessor
        new_output_batch.counts = self.counts + output_batch.counts
        return new_output_batch

    def to_tensor(self, *args):
        raise NotImplementedError

    def __str__(self):
        return ("loss: %f"
             +"\nattention_entropy: %f"
             +"\ntraceback_attention_entropy: %f"
             +"\naccuracy: %f"
             +"\nmacro_averaged - p: %f, r: %f, f1: %f"
             +"\nmicro_averaged - p: %f, r: %f, f1: %f") % (
            self.batch_stats['loss']/self.batch_length,
            self.batch_stats['attention_entropy']/self.batch_length,
            self.batch_stats['traceback_attention_entropy']/self.batch_length,
            self.batch_stats['accuracy_sum']/self.batch_length,
            *precision_recall_f1(
                *list(self.counts),
                reduce='macro'
            ),
            *precision_recall_f1(
                *list(self.counts),
                reduce='micro'
            )
        )

    def write_to_tensorboard(self, writer, iterator_info):
        global_step = iterator_info.batches_seen
        writer.add_scalar('loss', self.batch_stats['loss']/self.batch_length, global_step)
        writer.add_scalar('attention_entropy', self.batch_stats['attention_entropy']/self.batch_length, global_step)
        writer.add_scalar('traceback_attention_entropy', self.batch_stats['traceback_attention_entropy']/self.batch_length, global_step)
        writer.add_scalar('accuracy_sum', self.batch_stats['accuracy_sum']/self.batch_length, global_step)
        macro_p, macro_r, macro_f1 = precision_recall_f1(
            *list(self.counts), reduce='macro')
        micro_p, micro_r, micro_f1 = precision_recall_f1(
            *list(self.counts), reduce='micro')
        writer.add_scalar('macro_metrics/precision', macro_p, global_step)
        writer.add_scalar('macro_metrics/recall', macro_r, global_step)
        writer.add_scalar('macro_metrics/f1', macro_f1, global_step)
        writer.add_scalar('micro_metrics/precision', micro_p, global_step)
        writer.add_scalar('micro_metrics/recall', micro_r, global_step)
        writer.add_scalar('micro_metrics/f1', micro_f1, global_step)
        nq = self.outputs['labels'].size(1)
        code_mask = (torch.arange(nq, device=self.outputs['labels'].device) < self.outputs['num_codes'].unsqueeze(1))
        labels = self.outputs['labels'][code_mask]
        scores = self.outputs['scores'][code_mask]
        writer.add_histogram('scores_all', scores, global_step)
        writer.add_histogram('scores_0', scores[labels==0], global_step)
        writer.add_histogram('scores_1', scores[labels==1], global_step)
        #writer.add_figure('category_scores', self.create_category_scores_figure(), global_step)

    def create_category_scores_figure(self):
        data = ((np.array([.1, .3, .2]), np.array([.01, .02, .03])),)
        return plot_stacked_bar(data, x_ticks=None, stack_labels=None, y_label=None, title=None, show_nums=None, y_lim=None, file=None, figsize=None)

    def write_results(self):
        pass

class OutputBatchTest(OutputBatch):
    @classmethod
    def from_outputs(cls, postprocessor, batch, outputs):
        summary_counts = postprocessor.add_summary_stats(batch, outputs)
        loss, stats = cls.get_loss_stats(batch, outputs)
        with torch.autograd.no_grad():
            return loss, cls(len(batch), stats, batch=batch, outputs=outputs, postprocessor=postprocessor, summary_counts=summary_counts)

    def __init__(self, *args, summary_counts=None, **kwargs):
        kwargs['outputs'] = None
        super(OutputBatchTest, self).__init__(*args, **kwargs)
        self.summary_counts = summary_counts

    def __add__(self, output_batch):
        new_output_batch = super(OutputBatchTest, self).__add__(output_batch)
        if self.summary_counts is not None:
            new_output_batch.summary_counts = self.summary_counts + output_batch.summary_counts
        else:
            new_output_batch.summary_counts = None
        return new_output_batch

    def __str__(self):
        original_string = super(OutputBatchTest, self).__str__()
        if self.summary_counts is not None:
            original_string += "\nsummary - p: %f, r: %f, f1: %f" %\
                precision_recall_f1(*list(self.summary_counts))
        return original_string

    def write_to_tensorboard(self, *args, **kwargs):
        return
        super(BatchInfoTest, self).write_to_tensorboard(*args, **kwargs)
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
    def test_func(cls, batch, total_num_codes, scores, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, codes=None, labels=None):
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'context_vec':context_vec, 'article_sentences_lengths':article_sentences_lengths,
                   'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports'],
                   'clustering':clustering}
        if labels is not None:
            stats = statistics_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels)
        else:
            stats = {}
        return results, stats

    def __init__(self, batch_length, batch_stats, **kwargs):
        kwargs['outputs'] = None
        super(OutputBatchApplications, self).__init__(batch_length, batch_stats, **kwargs)
        if 'results' in batch_stats.keys():
            self.results = batch_stats['results']
            del batch_stats['results']
