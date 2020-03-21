from pytt.testing.raw_individual_processor import RawIndividualProcessor
from model_loader import load_model_components


class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, postprocessor, batcher):
        super(GenericProcessor, self).__init__(model, postprocessor, batcher)

    def process_datapoint(self, reports, tag=None, description=None, description_linearization=None, label=None):
        raw_datapoint = {'reports':reports}
        if tag is not None:
            raw_datapoint['targets'] = [tag]
        if description is not None:
            raw_datapoint['descriptions'] = [description]
        if description_linearization is not None:
            raw_datapoint['description_linearizations'] = [description_linearization]
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, model_type, hierarchy, model_file=None, device='cpu', cluster=False, sentences_per_checkpoint=100):
        batcher, model, postprocessor = load_model_components(model_type, hierarchy, run_type='applications', model_file=model_file, device=device, cluster=cluster, sentences_per_checkpoint=sentences_per_checkpoint)
        super(DefaultProcessor, self).__init__(model, postprocessor, batcher)

    def takes_nl_queries(self):
        code_embedding_types = self.batcher.get_code_embedding_type_params().keys()
        return 'codes' not in code_embedding_types and 'linearized_codes' not in code_embedding_types
