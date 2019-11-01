import numpy as np
import torch
from pytt.utils import pad_and_concat
from pytt.iteration_info import BatchInfo as BI
from .model import statistics_func

def precision_recall_f1(true_positives, positives, relevants, reduce='macro'):
    if reduce == 'micro':
        true_positives, positives, relevants = true_positives.sum(), positives.sum(), relevants.sum()
    precision = (true_positives/positives).masked_fill(positives == 0, 0)
    recall = (true_positives/relevants).masked_fill(relevants == 0, 0)
    f1 = (2*precision*recall/(precision + recall)).masked_fill((precision+recall) == 0, 0)
    if reduce == 'macro':
        precision, recall, f1 = precision.mean(), recall.mean(), f1.mean()
    return precision.item(), recall.item(), f1.item()

class BatchInfo(BI):
    def stats(self):
        returned_stats = statistics_func(**self.batch_outputs)
        self.counts = pad_and_concat([returned_stats['true_positives'],
                                      returned_stats['positives'],
                                      returned_stats['relevants']],
                                     auto=False)
        del returned_stats['true_positives']
        del returned_stats['positives']
        del returned_stats['relevants']
        return {
            k:v.item() for k,v in returned_stats.items()}

    def __add__(self, batch_info):
        new_batch_info = super(BatchInfo, self).__add__(batch_info)
        new_batch_info.counts = self.counts + batch_info.counts
        return new_batch_info

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

    def filter(self):
        self.batch_outputs = {
            'scores':self.batch_outputs['scores'],
            'labels':self.batch_outputs['labels'],
            'num_codes':self.batch_outputs['num_codes']}
        self.batch = None

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
        nq = self.batch_outputs['labels'].size(1)
        code_mask = (torch.arange(nq, device=self.batch_outputs['labels'].device) < self.batch_outputs['num_codes'].unsqueeze(1))
        labels = self.batch_outputs['labels'][code_mask]
        scores = self.batch_outputs['scores'][code_mask]
        writer.add_histogram('scores_all', scores, global_step)
        writer.add_histogram('scores_0', scores[labels==0], global_step)
        writer.add_histogram('scores_1', scores[labels==1], global_step)


def get_batch_info_class(loss_func):
    class BatchInfoTest(BatchInfo):
        def stats(self):
            results = super(BatchInfoTest, self).stats()
            return {'loss':loss_func(**self.batch_outputs), **results}
    return BatchInfoTest
