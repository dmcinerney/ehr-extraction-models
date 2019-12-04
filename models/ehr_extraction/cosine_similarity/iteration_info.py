import numpy as np
import torch
from pytt.iteration_info import BatchInfo as BI
from .model import statistics_func


class BatchInfo(BI):
    def stats(self):
        return statistics_func(**self.batch_outputs)
