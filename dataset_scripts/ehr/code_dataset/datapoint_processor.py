import torch
from pytt.utils import read_pickle
from pytt.testing.raw_individual_processor import RawIndividualProcessor
from models.ehr_extraction.code_supervision.model import Model, loss_func_creator, statistics_func
from models.ehr_extraction.code_supervision.iteration_info import BatchInfo
from dataset_scripts.ehr.code_dataset.batcher import Batcher

loss_func = loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1)

class BatchInfoTest(BatchInfo):
    def stats(self):
        self.results, stats = test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

def test_func(batch, scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels=None):
    # TODO: make result from batch
    results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths}
    if labels is not None:
        loss = loss_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels)
        stats = statistics_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels)
        stats = {'loss': loss, **stats}
    else:
        stats = {}
    results['tokenized_text'] = batch.instances[0]['tokenized_sentences']
    return results, stats

class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher):
        super(GenericProcessor, self).__init__(model, batcher, batch_info_class=BatchInfoTest)

    def process_datapoint(self, reports_text, code, label=None):
        raw_datapoint = {'reports':reports_text, 'targets':[code]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, code_graph_file, model_file):
        batcher = Batcher(read_pickle(code_graph_file))
        model = Model(len(batcher.code_graph.nodes), sentences_per_checkpoint=17, device1='cuda:1', device2='cpu')
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.correct_devices()
        model.eval()
        super(DefaultProcessor, self).__init__(model, batcher)
