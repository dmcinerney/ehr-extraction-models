import os
from shutil import copyfile
import torch
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.training.tracker import Tracker
from pytt.distributed import distributed_wrapper
from pytt.logger import logger
from dataset import init_dataset
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from shutil import copyfile

data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded'
train_file = os.path.join(data_dir, 'train.data')
val_file = os.path.join(data_dir, 'val.data')
used_targets_file = os.path.join(data_dir, 'used_targets.txt')
#model_type = 'code_supervision_unfrozen'
#save_checkpoint_folder = 'checkpoints2/code_supervision_unfrozen2'
#load_checkpoint_folder = 'checkpoints2/code_supervision'
model_type = 'code_supervision_only_description'
save_checkpoint_folder = 'checkpoints3/code_supervision_only_description'
load_checkpoint_folder = None
device = 'cuda:0'

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    batch_size = 8
    epochs = 2
    train_dataset = init_dataset(train_file)
    val_dataset = init_dataset(val_file)
    if load_checkpoint_folder is None:
        indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=epochs)
        val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
        model_file, optimizer_file = None, None
    else:
        indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'train_indices_iterator.pkl'))
        indices_iterator.set_stop(epochs=epochs)
        val_indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'val_indices_iterator.pkl'))
        val_indices_iterator.set_stop(iterations=len(indices_iterator))
        model_file, optimizer_file = os.path.join(load_checkpoint_folder, 'model_state.tpkl'), os.path.join(load_checkpoint_folder, 'optimizer_state.tpkl')
    batcher, model, batch_info_class, optimizer, loss_func = load_model_components(model_type, device=device, model_file=model_file) #, optimizer_file=optimizer_file)
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=4) #, num_workers=4)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tracker = Tracker(checkpoint_folder=save_checkpoint_folder, checkpoint_every=10)
#    if load_checkpoint_folder is not None:
#        tracker.needs_graph = False
    tracker.needs_graph = False
    trainer = Trainer(model, optimizer, batch_iterator, val_iterator=val_iterator, val_every=10, batch_info_class=batch_info_class, tracker=tracker)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func)

if __name__ == '__main__':
    if os.path.exists(used_targets_file) and save_checkpoint_folder is not None:
        copyfile(used_targets_file, os.path.join(save_checkpoint_folder, 'used_targets.txt'))
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
