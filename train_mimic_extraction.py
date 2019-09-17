import torch
from pytorch_pretrained_bert.optimization import BertAdam
from pytt.utils import seed_state
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.logger import logger
from dataset import split_dataset
from dataset_scripts.ehr.batcher import EHRBatcher
from dataset_scripts.ehr.preprocess_codes import get_codes
from models.mimic_extraction.model import ClinicalBertMimicExtraction, loss_func, statistics_func
from models.mimic_extraction.iteration_info import BatchInfo

def get_optimizer(param_optimizer, num_train_steps):
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-3,
                         warmup=.1,
                         t_total=num_train_steps)
    return optimizer


def main(checkpoint_folder=None):
    seed_state()
    logger.set_verbosity(2)
    batch_size = 2
    device = 'cuda:1'
    train_dataset, val_dataset = split_dataset('data/mimic/mimic_reports_to_codes.data')
    codes = {code:i for i,code in enumerate(get_codes(train_dataset.df))}
    batcher = EHRBatcher(codes)
    indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=10)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=2, devices=device)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=2, devices=device)
    model = ClinicalBertMimicExtraction(len(codes)).to(device)
    optimizer = get_optimizer(list(model.named_parameters()), len(indices_iterator))
    #optimizer = torch.optim.Adam(list(model.parameters()))
    trainer = Trainer(model, optimizer, batch_iterator, checkpoint_folder='checkpoints/clinical_bert_mimic_extraction/checkpoint',
                      checkpoint_every=10, val_iterator=val_iterator, val_every=10, batch_info_class=BatchInfo)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func, statistics_func=statistics_func)

if __name__ == '__main__':
    main()
