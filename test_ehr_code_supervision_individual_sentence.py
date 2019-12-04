import os
import torch
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from dataset import init_dataset
from dataset_scripts.ehr.code_dataset.batcher import Batcher
from models.ehr_extraction.code_supervision_individual_sentence.model import Model, loss_func, statistics_func
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import get_batch_info_class
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP

val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes/val.data'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology.pkl'
load_checkpoint_folder = 'checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint'
#load_checkpoint_folder = None

BatchInfoTest = get_batch_info_class(loss_func)

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
    batch_size = 4
    device1 = 'cuda:%i' % (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
    device2 = 'cpu'
    val_dataset = init_dataset(val_file)
    batcher = Batcher(read_pickle(code_graph_file))
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    model = Model(len(batcher.code_graph.nodes), sentences_per_checkpoint=17, device1=device1, device2=device2, freeze_bert=True, reduce_code_embeddings=False)
    if load_checkpoint_folder is not None:
        model.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'model_state.tpkl'), map_location='cpu'))
    model.correct_devices()
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    model.eval()
    tester = Tester(model, val_iterator, batch_info_class=BatchInfoTest)
#    tester = Tester(model, val_iterator, batch_info_class=BatchInfoTest, tensorboard_dir=os.path.join(load_checkpoint_folder, 'tensorboard/test'))
    total_batch_info = tester.test()
    with open(os.path.join(load_checkpoint_folder, 'scores.txt'), 'w') as f:
        f.write(str(total_batch_info))

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
