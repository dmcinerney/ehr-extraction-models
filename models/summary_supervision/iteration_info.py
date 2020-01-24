from pytt.iteration_info import BatchInfo as BI
from .model import statistics_func

class BatchInfo(BI):
    def stats(self):
        return {
            k:v.item() for k,v in statistics_func(**self.batch_outputs).items()}
