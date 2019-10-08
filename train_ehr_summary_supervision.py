import pandas as pd
import torch
from pytt.utils import seed_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.logger import logger
from dataset import Dataset
from dataset_scripts.ehr.summary_dataset.batcher import Batcher
from models.ehr_extraction.summary_supervision.model import Model, loss_func, statistics_func
from pytt.distributed import distributed_wrapper
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP

train_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_to_seq/train.data'
val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_to_seq/val.data'
save_checkpoint_folder = 'checkpoints/ehr_extraction_summary_supervision/checkpoint'
load_checkpoint_folder = None

def main(load_checkpoint_folder=None):
    seed_state()
    logger.set_verbosity(2)
    batch_size = 4
#    device = 'cuda:%i' % (torch.distributed.get_rank() if torch.distributed.is_initialized() else 1)
    device = 'cpu'
    train_dataset = Dataset(pd.read_csv(train_file, compression='gzip'))
    val_dataset = Dataset(pd.read_csv(val_file, compression='gzip'))
    import pdb; pdb.set_trace()
    batcher = Batcher()
    indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=5)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=4, devices=device)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4, devices=device)
    model = Model(sentences_per_checkpoint=1).to(device)
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()))
    trainer = Trainer(model, optimizer, batch_iterator, checkpoint_folder=save_checkpoint_folder,
                      checkpoint_every=10, val_iterator=val_iterator, val_every=10)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func, statistics_func=statistics_func)

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
