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
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_linearization': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_linearization_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_description': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_description=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_only_description_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_description=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB if run_type == 'training' else (OBT if run_type == 'testing' else OBA)))},
    'code_supervision_individual_sentence': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher:Model_is(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device1=device, device2='cpu'),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, (OB_cs if run_type == 'training' else (OBT_cs if run_type == 'testing' else OBA_cs)))},
    'cosine_similarity': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True, add_special_tokens=False),
        'model_class': lambda device, batcher:Model_cs(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device=device),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, OBA_cs)},
    'distance': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True, add_special_tokens=False),
        'model_class': lambda device, batcher:Model_d(sentences_per_checkpoint=p.sentences_per_checkpoint, num_codes=len(batcher.code_idxs), device=device),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, OBA_cs)},
    'tfidf_similarity': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True, tfidf_tokenizer=True),
        'model_class': lambda device, batcher: Model_tfidf(device=device),
        'postprocessor': lambda batcher, run_type: Postprocessor(batcher.code_idxs, OBA_cs)},
}


def load_model_components(model_name, code_graph_file, run_type='training', device='cpu', model_file=None, optimizer_file=None):
    batcher = model_components[model_name]['batcher_class'](read_pickle(code_graph_file), run_type)
    model = model_components[model_name]['model_class'](device, batcher)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
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
