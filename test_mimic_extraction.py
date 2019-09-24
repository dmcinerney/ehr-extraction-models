import torch
from pytt.utils import seed_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.testing.tester import Tester
from pytt.logger import logger
from dataset import init_dataset
from dataset_scripts.ehr.batcher import EHRBatcher
from models.ehr_extraction.model import ClinicalBertExtraction, loss_func, statistics_func
from models.ehr_extraction.iteration_info import BatchInfo

def test_func(*args, **kwargs):
    return None, {'loss':loss_func(*args, **kwargs),
            **statistics_func(*args, **kwargs)}

def main(checkpoint_folder=None):
    seed_state()
    logger.set_verbosity(2)
    batch_size = 2
    device = 'cuda:1'
    val_dataset = init_dataset('data/mimic/val_mimic.data')
    codes = {code:i for i,code in enumerate(read_pickle('data/mimic/codes.pkl'))}
    batcher = EHRBatcher(codes)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=2, devices=device)
    model = ClinicalBertExtraction(len(codes))
    model.load_state_dict(torch.load("checkpoints/clinical_bert_mimic_extraction/checkpoint/model_state.tpkl", map_location=device))
    model.eval()
    tester = Tester(model, val_iterator, batch_info_class=BatchInfo)
    tester.test(test_func)

if __name__ == '__main__':
    main()
