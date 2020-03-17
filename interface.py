import os
from datapoint_processor import DefaultProcessor
from processing.batcher import Batcher
from pytt.utils import read_pickle
from utils import get_queries
from hierarchy import Hierarchy

codes_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl' # hack to create batcher (it is not actually used because batcher does not return anything code-related)
model_info = {
    'cosine_similarity': ('cosine_similarity', None, ['description']),
    'distance': ('distance', None, ['description']),
    'tfidf_similarity': ('tfidf_similarity', None, ['description']),
    'code_supervision_unfrozen': ('code_supervision', '/home/jered/Documents/projects/ehr-extraction-models/bwh_models/code_supervision_unfrozen', ['tag']),
    'code_supervision_only_description_unfrozen': ('code_supervision_only_description_unfrozen', '/home/jered/Documents/projects/ehr-extraction-models/bwh_models/code_supervision_only_description_unfrozen', ['description']),
    'code_supervision_only_linearization_description_unfrozen': ('code_supervision_only_linearization_description_unfrozen', '/home/jered/Documents/projects/ehr-extraction-models/bwh_models/code_supervision_only_linearization_description_unfrozen', ['description_linearization']),
}

class TokenizerInterface:
    def __init__(self):
        self.hierarchy = Hierarchy.from_graph(read_pickle(codes_file))
        self.batcher = Batcher(self.hierarchy)
        self.linearizations = {n:self.hierarchy.linearize(n) for n in self.hierarchy.descriptions.keys()}

    def get_hierarchy(self):
        return self.hierarchy.to_dict()

    def tokenize(self, reports, num_sentences='default'):
        raw_datapoint = {'reports':reports, 'targets':[next(iter(self.hierarchy.descriptions.keys()))]}
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
                model_info[k][0],
                Hierarchy.from_dict(read_pickle(os.path.join(model_info[k][1], 'hierarchy.pkl')))\
                    if model_info[k][1] is not None else self.hierarchy,
                model_file=os.path.join(model_info[k][1], 'model_state.tpkl') if model_info[k][1] is not None else None,
                device=device,
                cluster=True)
            for k in self.models}
        self.trained_queries = {k:get_queries(os.path.join(model_info[k][1], 'used_targets.txt'))
                                if model_info[k][1] is not None else list(self.hierarchy.descriptions.keys())
                                for k in self.models}

    def get_trained_queries(self, model):
        return self.trained_queries[model]

    def with_custom(self, model):
        return self.dps[model].takes_nl_queries()

    def get_models(self):
        return self.models

    def query_reports(self, model, reports, tag=None, description=None, description_linearization=None):
        query_kwargs = {'tag':tag, 'description':description,'description_linearization':description_linearization}
        query_kwargs = {cet:query_kwargs[cet] for cet in model_info[model][2]}
        results = self.dps[model].process_datapoint(reports, **query_kwargs).results
        attention = results['attention'][0,0]
        # TODO: need to adjust this to cover if attention is all zero
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
            'clustering':results['clustering'][0][0],
        }
        if 'scores' in results.keys():
            return_dict['score'] = results['scores'].item()
        return return_dict
