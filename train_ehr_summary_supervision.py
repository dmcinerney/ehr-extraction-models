import os
import torch
import pandas as pd
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.training.tracker import Tracker
from pytt.distributed import distributed_wrapper
from pytt.logger import logger
from dataset import Dataset
from dataset_scripts.ehr.summary_dataset.batcher import SupervisedBatcher
from models.ehr_extraction.summary_supervision.model import Model, loss_func
from models.ehr_extraction.summary_supervision.iteration_info import BatchInfo
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
from utils import set_dropout

train_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_to_seq/train.data'
val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_to_seq/val.data'
save_checkpoint_folder = 'checkpoints/ehr_extraction_summary_supervision/checkpoint2'
#save_checkpoint_folder = None
load_checkpoint_folder = 'checkpoints/ehr_extraction_summary_supervision/checkpoint2'
#load_checkpoint_folder = None

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    batch_size = 4
    epochs = 1
    device1 = 'cuda:%i' % (torch.distributed.get_rank() if torch.distributed.is_initialized() else 1)
    device2 = 'cpu'
    df = pd.read_csv(train_file, compression='gzip', nrows=50000)
    train_dataset = Dataset(df[df.impression == df.impression])
    df = pd.read_csv(val_file, compression='gzip')
    val_dataset = Dataset(df[df.impression == df.impression])
    batcher = SupervisedBatcher()
    if load_checkpoint_folder is None:
        indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=epochs)
        val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    else:
        indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'train_indices_iterator.pkl'))
        indices_iterator.set_stop(epochs=epochs)
        val_indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'val_indices_iterator.pkl'))
        val_indices_iterator.set_stop(iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=4)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    start_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.cls_token])[0]
    stop_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.sep_token])[0]
    mask_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.mask_token])[0]
    model = Model({'start':start_token_id, 'stop':stop_token_id, 'mask':mask_token_id}, batcher.tokenizer.vocab_size, sentences_per_checkpoint=10, device1=device1, device2=device2, freeze_bert=True)
    if load_checkpoint_folder is not None:
        model.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'model_state.tpkl'), map_location='cpu'))
    model.correct_devices()
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()))
    if load_checkpoint_folder is not None:
        optimizer.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'optimizer_state.tpkl')))
    tracker = Tracker(checkpoint_folder=save_checkpoint_folder)
    tracker.needs_graph = False
    trainer = Trainer(model, optimizer, batch_iterator, checkpoint_folder=save_checkpoint_folder,
                      checkpoint_every=10, val_iterator=val_iterator, val_every=10,
                      tracker=tracker, batch_info_class=BatchInfo)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func)

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
