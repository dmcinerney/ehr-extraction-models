import os
from shutil import copyfile
import torch
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.training.tracker import Tracker
from pytt.distributed import distributed_wrapper
from pytt.logger import logger
from processing.dataset import init_dataset
#from fairseq.legacy_distributed_data_parallel\
#        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from shutil import copyfile
from argparse import ArgumentParser
import parameters as p

def main(model_type, train_file, code_graph_file, val_file=None, save_checkpoint_folder=None, load_checkpoint_folder=None, device='cuda:0',
         batch_size=p.batch_size, epochs=p.epochs, limit_rows_train=p.limit_rows_train, limit_rows_val=p.limit_rows_val, subbatches=p.subbatches,
         num_workers=p.num_workers, checkpoint_every=p.checkpoint_every, val_every=p.val_every):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    train_dataset = init_dataset(train_file, limit_rows=limit_rows_train)
    if val_file is not None:
        val_dataset = init_dataset(val_file, limit_rows=limit_rows_val)
    if load_checkpoint_folder is None:
        indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=epochs)
        if val_file is not None:
            val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
        model_file, optimizer_file = None, None
    else:
        indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'train_indices_iterator.pkl'))
        indices_iterator.set_stop(epochs=epochs)
        if val_file is not None:
            val_indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'val_indices_iterator.pkl'))
            val_indices_iterator.set_stop(iterations=len(indices_iterator))
        model_file, optimizer_file = os.path.join(load_checkpoint_folder, 'model_state.tpkl'), os.path.join(load_checkpoint_folder, 'optimizer_state.tpkl')
    batcher, model, postprocessor, optimizer = load_model_components(model_type, code_graph_file, device=device, model_file=model_file, optimizer_file=optimizer_file)
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=subbatches, num_workers=num_workers)
    if val_file is not None:
        val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=subbatches)
    else:
        val_iterator = None
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tracker = Tracker(checkpoint_folder=save_checkpoint_folder, checkpoint_every=checkpoint_every)
#    if load_checkpoint_folder is not None:
#        tracker.needs_graph = False
    tracker.needs_graph = False
    trainer = Trainer(model, postprocessor, optimizer, batch_iterator, val_iterator=val_iterator, val_every=val_every, tracker=tracker)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("--data_dir", default=p.data_dir)
    parser.add_argument("--code_graph_file", default=p.code_graph_file)
    parser.add_argument("--save_checkpoint_folder", default=None)
    parser.add_argument("--load_checkpoint_folder", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    train_file = os.path.join(args.data_dir, 'train.data')
    val_file = os.path.join(args.data_dir, 'val.data')
    counts_file = os.path.join(args.data_dir, 'counts.pkl')
    used_targets_file = os.path.join(args.data_dir, 'used_targets.txt')

    if args.save_checkpoint_folder is not None:
        copyfile(args.code_graph_file, os.path.join(args.save_checkpoint_folder, 'code_graph.pkl'))
        if os.path.exists(counts_file):
            copyfile(counts_file, os.path.join(args.save_checkpoint_folder, 'counts.pkl'))
        if os.path.exists(used_targets_file):
            copyfile(used_targets_file, os.path.join(args.save_checkpoint_folder, 'used_targets.txt'))
    main(args.model_type, train_file, args.code_graph_file, val_file=val_file,
         save_checkpoint_folder=args.save_checkpoint_folder, load_checkpoint_folder=args.load_checkpoint_folder,
         device=args.device)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(args.model_type, train_file, args.code_graph_file, val_file=val_file,
#         save_checkpoint_folder=args.save_checkpoint_folder, load_checkpoint_folder=args.load_checkpoint_folder,
#         device=args.device)
