import os
import torch
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from dataset import init_dataset
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components

val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded/val.data'
model_type = 'code_supervision_only_linearization'
load_checkpoint_folder = 'checkpoints/code_supervision_only_linearization'
device = 'cuda:1'

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    batch_size = 8
    val_dataset = init_dataset(val_file)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    model_file = None if load_checkpoint_folder is None else os.path.join(load_checkpoint_folder, 'model_state.tpkl')
    code_graph_file = None if load_checkpoint_folder is None else os.path.join(load_checkpoint_folder, 'code_graph.pkl')
    batcher, model, batch_info_class = load_model_components(model_type, code_graph_file, run_type='testing', device=device, model_file=model_file)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tester = Tester(model, val_iterator, batch_info_class=batch_info_class)
#    tester = Tester(model, val_iterator, batch_info_class=batch_info_class, tensorboard_dir=os.path.join(load_checkpoint_folder, 'tensorboard/test'))
    total_batch_info = tester.test()
    with open(os.path.join(load_checkpoint_folder, 'scores.txt'), 'w') as f:
        f.write(str(total_batch_info))

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
