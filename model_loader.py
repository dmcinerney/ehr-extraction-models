import torch
from batcher import Batcher
from models.code_supervision.model import Model, loss_func_creator
from models.code_supervision.iteration_info import BatchInfo as BI, create_batch_info_test as create_BIT, create_batch_info_applications as create_BIA
from models.code_supervision_individual_sentence.model import Model as Model_is, loss_func as loss_func_is
from models.code_supervision_individual_sentence.iteration_info import BatchInfo as BI_is, BatchInfoTest as BIT_is, BatchInfoApplications as BIA_is
from models.cosine_similarity.model import Model as Model_cs
from models.cosine_similarity.iteration_info import BatchInfoApplications as BIA_cs
from models.distance.model import Model as Model_d
from models.tfidf_similarity.model import Model as Model_tfidf
from pytt.utils import read_pickle

model_components = {
    'code_supervision': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_id=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_only_linearization': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_only_linearization_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_linearization=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_linearization_embeddings=batcher.graph_ops.max_index+1, device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_only_description': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, sample_top=100 if run_type == 'training' else None, code_description=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=True),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_only_description_unfrozen': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher:Model(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu', freeze_bert=False, dropout=0),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.00001),
        'loss_func': loss_func_creator(),
        'batch_info_classes': {
            'training': BI,
            'testing': create_BIT(loss_func_creator()),
            'applications': create_BIA(loss_func_creator())}},
    'code_supervision_individual_sentence': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_id=True),
        'model_class': lambda device, batcher:Model_is(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device1=device, device2='cpu'),
        'optimizer_class': lambda parameters: torch.optim.Adam(parameters, lr=.001),
        'loss_func': loss_func_is,
        'batch_info_classes': {
            'training': BI_is,
            'testing': BIT_is,
            'applications': BIA_is}},
    'cosine_similarity': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher:Model_cs(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device=device),
        'batch_info_classes': {
            'applications':BIA_cs}},
    'distance': {
        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True),
        'model_class': lambda device, batcher:Model_d(sentences_per_checkpoint=17, num_codes=len(batcher.code_idxs), device=device),
        'batch_info_classes': {
            'applications':BIA_cs}},
# TODO CHARLIE: uncomment this
#    'tfidf_similarity': {
#        'batcher_class': lambda code_graph, run_type: Batcher(code_graph, code_description=True, tfidf_tokenizer=True),
#        'model_class': lambda device, batcher: Model_tfidf(), # it's weird, but you can just ignore the arguments to this lambda function
#        'batch_info_classes': {
#            'applications':BIA_cs}},
}


def load_model_components(model_name, code_graph_file, run_type='training', device='cpu', model_file=None, optimizer_file=None):
    batcher = model_components[model_name]['batcher_class'](read_pickle(code_graph_file), run_type)
    model = model_components[model_name]['model_class'](device, batcher)
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
