import torch
from dataset_scripts.ehr.code_dataset.batcher import Batcher
from dataset_scripts.ehr.code_dataset.batch_info_test_fordp import BatchInfoTest as BIT_fordp, BatchInfoTest_is as BIT_is_fordp, BatchInfoTest_cs as BIT_cs_fordp
from models.ehr_extraction.code_supervision.model import Model, loss_func_creator
from models.ehr_extraction.code_supervision.iteration_info import BatchInfo as BI, get_batch_info_test_class
loss_func = loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=.25)
BIT = get_batch_info_test_class(loss_func)
from models.ehr_extraction.code_supervision_individual_sentence.model import Model as Model_is, loss_func as loss_func_is
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import BatchInfo as BI_is, get_batch_info_test_class
BIT_is = get_batch_info_test_class(loss_func_is)
from models.ehr_extraction.cosine_similarity.model import Model as Model_cs
from pytt.utils import read_pickle

model_components = {
    'code_supervision': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_only_linearization': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_only_linearization_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_with_description': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_id=True, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True, num_code_embedding_types=2),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_with_description_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_id=True, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, num_code_embedding_types=2, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_only_description': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_only_description_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func,
        'batch_info_classes': {
            'training': BI,
            'testing': BIT,
            'applications': BIT_fordp}},
    'code_supervision_individual_sentence': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_id=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model_is(*args, **kwargs, num_codes=len(batcher.code_idxs), device1=device, device2='cpu'),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func_is,
        'batch_info_classes': {
            'training': BI_is,
            'testing': BIT_is,
            'applications': BIT_is_fordp}},
    'cosine_similarity': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model_cs(*args, **kwargs, num_codes=len(batcher.code_idxs), device=device),
        'batch_info_classes': {
            'applications':BIT_cs_fordp}},
    'distance': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher, *args, **kwargs:Model_cs(*args, **kwargs, num_codes=len(batcher.code_idxs), device=device),
        'batch_info_classes': {
            'applications':BIT_cs_fordp}},
}

def load_model_components(model_name, code_graph_file, run_type='training', device='cpu', model_file=None, optimizer_file=None):
    batcher = model_components[model_name]['batcher_class'](read_pickle(code_graph_file), run_type)
    model = model_components[model_name]['model_class'](device, batcher, sentences_per_checkpoint=17)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.correct_devices()
    batch_info_class = model_components[model_name]['batch_info_classes'][run_type]
    if run_type == 'training':
        model.train()
        optimizer = model_components[model_name]['optimizer_class'](list(model.parameters()))
        if optimizer_file is not None:
            optimizer.load_state_dict(torch.load(optimizer_file))
        loss_func = model_components[model_name]['loss_func']
        return batcher, model, batch_info_class, optimizer, loss_func
    else:
        model.eval()
        return batcher, model, batch_info_class
