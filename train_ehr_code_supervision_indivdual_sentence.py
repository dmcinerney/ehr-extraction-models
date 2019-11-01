import os
import torch
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.training.tracker import Tracker
from pytt.distributed import distributed_wrapper
from pytt.logger import logger
from dataset import init_dataset
from dataset_scripts.ehr.code_dataset.batcher import Batcher
from models.ehr_extraction.code_supervision_individual_sentence.model import Model, loss_func
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import BatchInfo
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP

train_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes/train.data'
val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes/val.data'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology.pkl'
save_checkpoint_folder = 'checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint'
#save_checkpoint_folder = None
#load_checkpoint_folder = 'checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint'
load_checkpoint_folder = None

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    batch_size = 8
    epochs = 1
    device1 = 'cuda:%i' % (torch.distributed.get_rank() if torch.distributed.is_initialized() else 1)
    device2 = 'cpu'
    train_dataset = init_dataset(train_file)
    val_dataset = init_dataset(val_file)
    batcher = Batcher(read_pickle(code_graph_file))
    if load_checkpoint_folder is None:
        indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=epochs)
        val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    else:
        indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'train_indices_iterator.pkl'))
        indices_iterator.set_stop(epochs=epochs)
        val_indices_iterator = read_pickle(os.path.join(load_checkpoint_folder, 'val_indices_iterator.pkl'))
        val_indices_iterator.set_stop(iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=4, num_workers=4)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    model = Model(len(batcher.code_graph.nodes), sentences_per_checkpoint=50, device1=device1, device2=device2, freeze_bert=True)
    if load_checkpoint_folder is not None:
        model.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'model_state.tpkl'), map_location='cpu'))
    model.correct_devices()
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()))
    if load_checkpoint_folder is not None:
        optimizer.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'optimizer_state.tpkl')))
    tracker = Tracker(checkpoint_folder=save_checkpoint_folder, checkpoint_every=10)
    tracker.needs_graph = False
    trainer = Trainer(model, optimizer, batch_iterator, val_iterator=val_iterator, val_every=10, batch_info_class=BatchInfo, tracker=tracker)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func)

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
