import torch
from pytt.batching.postprocessor import StandardPostprocessor

class Postprocessor(StandardPostprocessor):
    def __init__(self, code_idxs, output_batch_class):
        self.code_idxs = code_idxs
        self.output_batch_class = output_batch_class

    def output_batch(self, batch, outputs):
        outputs['total_num_codes'] = torch.tensor(len(self.code_idxs))
        return self.output_batch_class.from_outputs(self.code_idxs, batch, outputs)
