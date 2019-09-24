import numpy as np
from pytt.iteration_info import BatchInfo as BI

def precision_recall_f1(true_positives, positives, relevants):
    precision = true_positives/positives if positives > 0 else 0
    recall = true_positives/relevants if relevants > 0 else 0
    f1 = 2*precision*recall/(precision + recall) if precision+recall > 0 else 0
    return precision, recall, f1

class BatchInfo(BI):
    def __str__(self):
        return "loss: %f\naccuracy: %f\np: %f, r: %f, f1: %f" % (
            self.batch_info_dict['loss']/self.batch_info_dict['_batch_length'],
            self.batch_info_dict['true_positives']/self.batch_info_dict['total_predicted'],
            *precision_recall_f1(
                self.batch_info_dict['true_positives'],
                self.batch_info_dict['positives'],
                self.batch_info_dict['relevants'],
            )
        )
