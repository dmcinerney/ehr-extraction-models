import numpy as np
from pytt.iteration_info import BatchInfo as BI

def precision_recall_f1(true_positives, positives, relevants):
    precision = true_positives/positives if positives > 0 else 0
    recall = true_positives/relevants if relevants > 0 else 0
    f1 = 2*precision*recall/(precision + recall) if precision+recall > 0 else 0
    return precision, recall, f1

class BatchInfo(BI):
    def __str__(self):
        averaged_scores = self.get_averaged_scores()
        step_info = ""
        step_info += "attention_overlap: "\
                     +str(averaged_scores["attention_overlap"])
        step_info += "\nlikelihood: "+str(averaged_scores["likelihood"])
        for i in range(3):
            class_i = averaged_scores["class_"+str(i)]
            step_info += "\nClass %i: p: %f, r: %f, f1: %f" % (
                i, class_i["p"], class_i["r"], class_i["f1"])
        ma = averaged_scores["macro_averaged"]
        step_info += "\nMacro Averaged: p: %f, r: %f, f1: %f" % (
            ma["p"], ma["r"], ma["f1"])
        return step_info

    def get_averaged_scores(self):
        averaged_scores = {}
        num_instances = self.batch_info_dict['_batch_length']
        averaged_scores["attention_overlap"] = \
            self.batch_info_dict["attention_overlap"]/num_instances
        averaged_scores["likelihood"] = \
            self.batch_info_dict["likelihood"]/num_instances
        prf1s = []
        for i in range(3):
            prf1s.append(precision_recall_f1(
                self.batch_info_dict["true_positives_"+str(i)],
                self.batch_info_dict["positives_"+str(i)],
                self.batch_info_dict["relevants_"+str(i)]
            ))
            averaged_scores["class_"+str(i)] = {
                k:v for k,v in zip(("p","r","f1"), prf1s[-1])}
        prf1_tot = np.array(prf1s).mean(0).tolist()
        averaged_scores["macro_averaged"] = {
            k:v for k,v in zip(("p","r","f1"), prf1_tot)}
        return averaged_scores

    def get_score(self):
        return self.get_averaged_scores()["macro_averaged"]["f1"]
