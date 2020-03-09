from pytt.testing.raw_individual_processor import RawIndividualProcessor
from model_loader import load_model_components


class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, postprocessor, batcher):
        super(GenericProcessor, self).__init__(model, postprocessor, batcher)

    def process_datapoint(self, reports, query, label=None, is_nl=False):
        raw_datapoint = {'reports':reports, 'queries' if is_nl else 'targets':[query]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, model_type, hierarchy, model_file=None, device='cpu'):
        batcher, model, postprocessor = load_model_components(model_type, hierarchy, run_type='applications', model_file=model_file, device=device)
        super(DefaultProcessor, self).__init__(model, postprocessor, batcher)

    def takes_nl_queries(self):
        return self.batcher.code_description and not self.batcher.code_id and not self.batcher.code_linearization
