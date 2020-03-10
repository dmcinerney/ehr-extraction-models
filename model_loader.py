import torch
from processing.batcher import Batcher
from processing.postprocessor import Postprocessor
from models.code_supervision.model import Model
from models.code_supervision.postprocessor import OutputBatch as OB, OutputBatchTest as OBT, OutputBatchApplications as OBA
from models.code_supervision_individual_sentence.model import Model as Model_is
from models.code_supervision_individual_sentence.postprocessor import OutputBatch as OB_is, OutputBatchTest as OBT_is, OutputBatchApplications as OBA_is
from models.cosine_similarity.model import Model as Model_cs
from models.cosine_similarity.postprocessor import OutputBatchApplications as OBA_cs
from models.distance.model import Model as Model_d
from models.tfidf_similarity.model import Model as Model_tfidf
from pytt.utils import read_pickle
import parameters as p

model_components = {
    'code_supervision': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_id=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_unfrozen': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_id=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_linearization': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_linearization=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_linearization_embeddings=batcher.hierarchy.indices["max_index"]+1, device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_linearization_unfrozen': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_linearization=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_linearization_embeddings=batcher.hierarchy.indices["max_index"]+1, device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_description': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_description=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_description_unfrozen': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_description=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_linearization_description_unfrozen': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, description_linearization=True),
        'model_class': lambda device, batcher, cluster:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_individual_sentence': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_id=True),
        'model_class': lambda device, batcher, cluster:Model_is(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types()),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB_is if run_type == 'training' else (OBT_is if run_type == 'testing' else OBA_is)))},
    'code_supervision_individual_sentence_unfrozen': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, resample_neg_proportion=.01 if run_type == 'training' else None, counts=counts, code_id=True),
        'model_class': lambda device, batcher, cluster:Model_is(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', cluster=cluster, code_embedding_types=batcher.get_code_embedding_types(), freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, (OB_is if run_type == 'training' else (OBT_is if run_type == 'testing' else OBA_is)))},
    'cosine_similarity': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, code_description=True, add_special_tokens=False),
        'model_class': lambda device, batcher, cluster:Model_cs(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device=device, cluster=cluster),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, OBA_cs)},
    'distance': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, code_description=True, add_special_tokens=False),
        'model_class': lambda device, batcher, cluster:Model_d(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device=device, cluster=cluster),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, OBA_cs)},
    'tfidf_similarity': {
        'batcher_class': lambda hierarchy, counts, run_type: Batcher(hierarchy, code_description=True, tfidf_tokenizer=True),
        'model_class': lambda device, batcher, cluster: Model_tfidf(device=device, cluster=cluster),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.hierarchy, batcher.code_idxs, OBA_cs)},
}


def load_model_components(model_name, hierarchy, run_type='training', device='cpu', model_file=None, optimizer_file=None, counts_file=None, cluster=False):
    counts = None if counts_file is None else read_pickle(counts_file)
    batcher = model_components[model_name]['batcher_class'](hierarchy, counts, run_type)
    model = model_components[model_name]['model_class'](device, batcher, cluster)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False) # TODO: take out strict=False
    model.correct_devices()
    postprocessor = model_components[model_name]['postprocessor'](batcher, run_type)
    if run_type == 'training':
        model.train()
        optimizer = model_components[model_name]['optimizer_class'](list(model.parameters()))
        if optimizer_file is not None:
            optimizer.load_state_dict(torch.load(optimizer_file))
        return batcher, model, postprocessor, optimizer
    else:
        model.eval()
        return batcher, model, postprocessor
