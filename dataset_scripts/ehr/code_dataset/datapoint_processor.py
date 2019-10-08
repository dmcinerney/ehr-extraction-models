import torch
from pytt.utils import read_pickle
from pytt.testing.raw_individual_processor import RawIndividualProcessor
from models.ehr_extraction.code_supervision.model import Model, loss_func, statistics_func
from dataset_scripts.ehr.code_dataset.batcher import Batcher


class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, test_func, device=None):
        super(GenericProcessor, self).__init__(model, batcher, test_func, device=device)

    def process_datapoint(self, reports_text, code, label=None):
        raw_datapoint = {'reports':reports_text, 'targets':[code]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, code_graph_file, model_file):
        device = 'cuda:1'
        batcher = Batcher(read_pickle(code_graph_file))
        model = Model(len(batcher.code_graph.nodes)).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        super(DefaultProcessor, self).__init__(model, batcher, self.test_func, device=device)

    def test_func(self, scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels=None):
        # TODO: make result from batch
        result = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths}
        if labels is not None:
            loss = loss_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels)
            stats = statistics_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels)
            stats = {'loss': loss, **stats}
        else:
            stats = {}
        return result, stats

    def process_datapoint(self, reports_text, code, label=None):
        batch, results, stats = super(DefaultProcessor, self).process_datapoint(reports_text, code, label=label)
        results['tokenized_text'] = batch.datapoints[0]['tokenized_sentences']
        return results, stats
