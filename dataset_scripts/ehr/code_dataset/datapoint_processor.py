from pytt.testing.raw_individual_processor import RawIndividualProcessor
from model_loader import load_model_components

model_files = {
    'code_supervision_with_description': '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/aligned/model_state.tpkl',
    'code_supervision_only_description': '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/description_only/model_state.tpkl',
    'code_supervision_individual_sentence': '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint/model_state.tpkl',
    'cosine_similarity': None,
}

class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, batch_info_class):
        super(GenericProcessor, self).__init__(model, batcher, batch_info_class=batch_info_class)

    def process_datapoint(self, reports, query, label=None, is_nl=False):
        raw_datapoint = {'reports':reports, 'queries' if is_nl else 'targets':[query]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, model_type):
        batcher, model, batch_info_class = load_model_components(model_type, run_type='applications', model_file=model_files[model_type], device='cuda:0')
        super(DefaultProcessor, self).__init__(model, batcher, batch_info_class)

    def takes_nl_queries(self):
        return self.batcher.instance_type == "only_description"
