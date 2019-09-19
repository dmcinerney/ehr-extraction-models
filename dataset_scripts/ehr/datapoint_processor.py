import pickle as pkl
from pytt.testing.raw_individual_processor import RawIndividualProcessor

class Processor(RawIndividualProcessor):
    def __init__(self, model, batcher, dataset_folder):
        super(Processor, self).__init__(model, batcher)
        with open(os.path.join(dataset_folder, 'codes.pkl'), 'rb') as f:
            self.codes = pkl.load(f)

    def process_datapoint(self, reports_text, code):
        # make datapoint here
        raw_datapoint = {'reports':text}
        super(Processor, self).process_datapoint(raw_datapoint)
