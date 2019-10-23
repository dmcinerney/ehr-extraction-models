import numpy as np
import torch
from pytt.iteration_info import BatchInfo as BI
from .model import statistics_func

def precision_recall_f1(true_positives, positives, relevants):
    precision = true_positives/positives if positives > 0 else 0
    recall = true_positives/relevants if relevants > 0 else 0
    f1 = 2*precision*recall/(precision + recall) if precision+recall > 0 else 0
    return precision, recall, f1

class BatchInfo(BI):
    def stats(self):
        return {
            k:v.item() for k,v in statistics_func(**self.batch_outputs).items()}

    def __str__(self):
        return "loss: %f\nattention_entropy: %f\ntraceback_attention_entropy: %f\naccuracy: %f\np: %f, r: %f, f1: %f" % (
            self.batch_stats['loss']/self.batch_length,
            self.batch_stats['attention_entropy']/self.batch_length,
            self.batch_stats['traceback_attention_entropy']/self.batch_length,
            self.batch_stats['accuracy_sum']/self.batch_length,
            *precision_recall_f1(
                self.batch_stats['true_positives'],
                self.batch_stats['positives'],
                self.batch_stats['relevants'],
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
        p, r, f1 = precision_recall_f1(
            self.batch_stats['true_positives'],
            self.batch_stats['positives'],
            self.batch_stats['relevants'])
        writer.add_scalar('metrics/precision', p, global_step)
        writer.add_scalar('metrics/recall', r, global_step)
        writer.add_scalar('metrics/f1', f1, global_step)
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
