from pytt.training.tracker import Tracker as OriginalTracker
from pytt.distributed import log_bool

class Tracker(OriginalTracker):
    def __init__(self, *args, expensive_val_every=None, expensive_val_func=None, **kwargs):
        super(Tracker, self).__init__(*args, **kwargs)
        self.expensive_val_every = expensive_val_every
        self.expensive_val_func = expensive_val_func

    def register_iteration(self, iteration_info, trainer):
        super(Tracker, self).register_iteration(iteration_info, trainer)
        if log_bool():
            if self.recurring_bool(iteration_info, self.expensive_val_every):
                self.expensive_val_func(iteration_info)

