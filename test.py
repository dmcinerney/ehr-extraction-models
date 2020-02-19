import os
import torch
from pytt.utils import seed_state, set_random_state, read_pickle, write_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from processing.dataset import init_dataset
#from fairseq.legacy_distributed_data_parallel\
#        import LegacyDistributedDataParallel as LDDP
from model_loader import load_model_components
from argparse import ArgumentParser
import parameters as p


def main(model_type, val_file, checkpoint_folder, device='cuda:0', batch_size=p.batch_size, subbatches=p.subbatches, num_workers=p.num_workers):
    if checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    val_dataset = init_dataset(val_file)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    model_file = os.path.join(checkpoint_folder, 'model_state.tpkl')
    code_graph_file = os.path.join(checkpoint_folder, 'code_graph.pkl')
    batcher, model, postprocessor = load_model_components(model_type, code_graph_file, run_type='testing', device=device, model_file=model_file)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=subbatches, num_workers=num_workers)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    tester = Tester(model, postprocessor, val_iterator)
#    tester = Tester(model, postprocessor, val_iterator, tensorboard_dir=os.path.join(load_checkpoint_folder, 'tensorboard/test'))
    postprocessor.add_output_dir(checkpoint_folder)
    total_output_batch = tester.test()
    with open(os.path.join(checkpoint_folder, 'scores.txt'), 'w') as f:
        f.write(str(total_output_batch))
    #total_output_batch.write_results()
    #write_pickle(postprocessor.summary_stats, os.path.join(checkpoint_folder, 'summary_stats.pkl'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("checkpoint_folder")
    parser.add_argument("--data_dir", default=p.data_dir)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    val_file = os.path.join(args.data_dir, 'val.data')

    main(args.model_type, val_file, args.checkpoint_folder, device=args.device)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(args.model_type, val_file, args.checkpoint_folder, device=args.device)
