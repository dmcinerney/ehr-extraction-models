import torch
from pytt.utils import read_pickle
from pytt.testing.raw_individual_processor import RawIndividualProcessor
from models.ehr_extraction.model import ClinicalBertExtraction, loss_func, statistics_func
from dataset_scripts.ehr.batcher import EHRBatcher


class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, test_func, device=None):
        super(GenericProcessor, self).__init__(model, batcher, test_func, device=device)

    def process_datapoint(self, reports_text, code, label=None):
        raw_datapoint = {'reports':reports_text, 'targets':[code]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, codes_file, model_file):
        device = 'cuda:0'
        codes = {code:i for i,code in enumerate(read_pickle(codes_file))}
        model = ClinicalBertExtraction(len(codes)).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        batcher = EHRBatcher(codes)
        super(DefaultProcessor, self).__init__(model, batcher, self.test_func, device=device)

    def test_func(self, scores, attention, traceback_attention, num_codes, labels=None):
        # TODO: make result from batch
        result = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention}
        if labels is not None:
            loss = loss_func(scores, attention, traceback_attention, num_codes, labels)
            stats = statistics_func(scores, attention, traceback_attention, num_codes, labels)
            stats = {'loss': loss, **stats}
        else:
            stats = {}
        return result, stats

    def process_datapoint(self, reports_text, code, label=None):
        batch, results, stats = super(DefaultProcessor, self).process_datapoint(reports_text, code, label=label)
        results['tokenized_text'] = batch.datapoints[0]['tokenized_sentences']
        return results, stats
