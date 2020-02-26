import os
from datapoint_processor import DefaultProcessor
from processing.batcher import Batcher
from pytt.utils import read_pickle
from utils import get_queries

codes_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl' # hack to create batcher (it is not actually used because batcher does not return anything code-related)
model_dirs = {
#    'code_supervision': ('code_supervision', '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/code_supervision'),
    'cosine_similarity': ('cosine_similarity', None),
    'distance': ('distance', None),
    'tfidf_similarity': ('tfidf_similarity', None),
    'code_supervision_only_description_unfrozen': ('code_supervision_only_description_unfrozen', '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/code_supervision_only_description_unfrozen'),
    'code_supervision_only_linearization_unfrozen': ('code_supervision_only_linearization_unfrozen', '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/code_supervision_only_linearization_unfrozen'),
}


class TokenizerInterface:
    def __init__(self):
        self.batcher = Batcher(read_pickle(codes_file))
        self.linearizations = {n:self.batcher.graph_ops.linearize(n) for n in self.batcher.graph_ops.graph.nodes}

    def get_descriptions(self):
        return {k:self.batcher.graph_ops.graph.nodes[k]['description']
                if self.batcher.graph_ops.graph.nodes[k]['description'] is not None else ''
                for k,v in self.batcher.code_idxs.items() if k != self.batcher.graph_ops.start_node}

    def get_hierarchy(self):
        return {"start":self.batcher.graph_ops.start_node,
                "options":self.batcher.graph_ops.node_idx_option,
                "indices":self.batcher.graph_ops.node_option_idx,
                "parents":{n:next(iter(self.batcher.graph_ops.graph.predecessors(n))) for n in self.batcher.graph_ops.graph.nodes if self.batcher.graph_ops.graph.in_degree(n) > 0}}

    def tokenize(self, reports, num_sentences='default'):
        raw_datapoint = {'reports':reports, 'targets':[next(iter(self.get_descriptions().keys()))]}
        if num_sentences != 'default':
           raw_datapoint['num_sentences'] = num_sentences
        instance = self.batcher.process_datapoint(raw_datapoint)
        sentence_spans = instance.sentence_spans
        original_reports = instance.raw_datapoint['reports']
        for i in range(len(sentence_spans)-1, -1, -1):
            if len(sentence_spans[i]) == 0:
                del sentence_spans[i]
                original_reports = original_reports.drop(original_reports.index[i])
        return {
            'tokenized_text':instance.tokenized_sentences,
            'sentence_spans':sentence_spans,
            'original_reports':original_reports
        }

class FullModelInterface(TokenizerInterface):
    def __init__(self, models_to_load=[], device='cpu'):
        super(FullModelInterface, self).__init__()
        self.models = models_to_load
        self.dps = {
            k:DefaultProcessor(
                model_dirs[k][0],
                os.path.join(model_dirs[k][1], 'code_graph.pkl') if model_dirs[k][1] is not None else codes_file,
                model_file=os.path.join(model_dirs[k][1], 'model_state.tpkl') if model_dirs[k][1] is not None else None,
                device=device)
            for k in self.models}
        self.trained_queries = {k:get_queries(os.path.join(model_dirs[k][1], 'used_targets.txt'))
                                if model_dirs[k][1] is not None else list(self.get_descriptions().keys())
                              for k in self.models}

    def get_trained_queries(self, model):
        return self.trained_queries[model]

    def with_custom(self, model):
        return self.dps[model].takes_nl_queries()

    def get_models(self):
        return self.models

    def query_reports(self, model, reports, query, is_nl=False):
        results = self.dps[model].process_datapoint(reports, query, is_nl=is_nl).results
        attention = results['attention'][0,0]
        min_avged = attention.sum(1, keepdim=True).min()/attention.size(1)
        sentence_level_attention = (attention-min_avged)/(attention.sum(1, keepdim=True).max()+.0001-min_avged)
        attention = (attention-attention.min())/(attention.max()-attention.min())
        traceback_attention = results['traceback_attention'][0,0]
        traceback_attention = (traceback_attention-traceback_attention.min())/(traceback_attention.max()-traceback_attention.min())
        return_dict = {
            'heatmaps':{
                'attention':[sent[:len(results['tokenized_text'][i])]
                    for i,sent in enumerate(attention.tolist())],
                'traceback_attention':[sent[:len(results['tokenized_text'][i])]
                    for i,sent in enumerate(traceback_attention.tolist())],
                'sentence_level_attention':[sent[:len(results['tokenized_text'][i])]
                    for i,sent in enumerate(sentence_level_attention.tolist())],
            },
        }
        if 'scores' in results.keys():
            return_dict['score'] = results['scores'].item()
        return return_dict
