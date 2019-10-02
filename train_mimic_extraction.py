import torch
from pytt.utils import seed_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.logger import logger
from dataset import init_dataset
from dataset_scripts.ehr.batcher import EHRBatcher
from models.ehr_extraction.model import ClinicalBertExtraction, loss_func, statistics_func
from models.ehr_extraction.iteration_info import BatchInfo

train_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes2/val_mimic.data'
val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes2/val_mimic.data'
codes_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes2/codes.pkl'
checkpoint_folder = 'checkpoints/clinical_bert_mimic_extraction/checkpoint'

def main(checkpoint_folder=None):
    seed_state()
    logger.set_verbosity(2)
    batch_size = 2
    device = 'cpu'
    train_dataset = init_dataset(train_file)
    val_dataset = init_dataset(val_file)
    codes = {code:i for i,code in enumerate(read_pickle(codes_file))}
    batcher = EHRBatcher(codes)
    indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=10)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=2, devices=device)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=2, devices=device)
    model = ClinicalBertExtraction(len(codes)).to(device)
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()))
    trainer = Trainer(model, optimizer, batch_iterator, checkpoint_folder=checkpoint_folder,
                      checkpoint_every=10, val_iterator=val_iterator, val_every=10, batch_info_class=BatchInfo)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func, statistics_func=statistics_func)

if __name__ == '__main__':
    main()
