from pytt.testing.raw_individual_processor import RawIndividualProcessor
from model_loader import load_model_components


class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, batch_info_class):
        super(GenericProcessor, self).__init__(model, batcher, batch_info_class=batch_info_class)

    def process_datapoint(self, reports, query, label=None, is_nl=False):
        raw_datapoint = {'reports':reports, 'queries' if is_nl else 'targets':[query]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, model_type, model_file):
        batcher, model, batch_info_class = load_model_components(model_type, run_type='applications', model_file=model_file, device='cuda:0')
        super(DefaultProcessor, self).__init__(model, batcher, batch_info_class)

    def takes_nl_queries(self):
        return self.batcher.instance_type == "only_description"
