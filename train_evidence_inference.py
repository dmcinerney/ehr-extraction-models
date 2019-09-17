import torch
from pytorch_pretrained_bert.optimization import BertAdam
from pytt.utils import seed_state
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.training.trainer import Trainer
from pytt.logger import logger
from dataset import init_dataset
from dataset_scripts.evidence_inference.batcher import EvidenceInferenceBatcher
#from dataset_scripts.pubmed.batcher import PubmedBatcher
from models.evidence_inference.model import ClinicalBertEvidenceInference, loss_func, statistics_func
from models.evidence_inference.iteration_info import BatchInfo

def get_optimizer(param_optimizer, num_train_steps):
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=.1,
                         t_total=num_train_steps)
    return optimizer


def main(checkpoint_folder=None):
    seed_state()
    logger.set_verbosity(2)
    batch_size = 8
    device = 'cuda:1'
#    train_dataset = init_dataset('data/pubmed/train_processed.data')
#    val_dataset = init_dataset('data/pubmed/val_processed.data')
    train_dataset = init_dataset('data/evidence_inference/train_processed.data')
    val_dataset = init_dataset('data/evidence_inference/val_processed.data')
    batcher = EvidenceInferenceBatcher()
    indices_iterator = init_indices_iterator(len(train_dataset), batch_size, random=True, epochs=10)
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size, random=True, iterations=len(indices_iterator))
    batch_iterator = batcher.batch_iterator(train_dataset, indices_iterator, subbatches=8, devices=device)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=8, devices=device)
    model = ClinicalBertEvidenceInference().to(device)
    optimizer = get_optimizer(list(model.named_parameters()), len(indices_iterator))
    trainer = Trainer(model, optimizer, batch_iterator, checkpoint_folder='checkpoints/clinical_bert_pubmed/checkpoint',
                      checkpoint_every=10, val_iterator=val_iterator, val_every=10, batch_info_class=BatchInfo)
    with torch.autograd.set_detect_anomaly(False):
        trainer.train(loss_func, statistics_func=statistics_func)

if __name__ == '__main__':
    main()
