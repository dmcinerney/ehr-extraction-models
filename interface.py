from dataset_scripts.ehr.code_dataset.datapoint_processor import DefaultProcessor
from dataset_scripts.ehr.code_dataset.batcher import Batcher
from pytt.utils import read_pickle

codes_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology.pkl'

class TokenizerInterface:
    def __init__(self):
        self.batcher = Batcher(read_pickle(codes_file))

    def get_queries(self):
        return {k:self.batcher.code_graph.nodes[k]['description']
                if 'description' in self.batcher.code_graph.nodes[k].keys() else ''
                for k,v in self.batcher.code_idxs.items()}

    def tokenize(self, reports):
        instance = self.batcher.process_datapoint({'reports':reports, 'targets':[next(iter(self.get_queries().keys()))]})
        return {
            'tokenized_text':instance.tokenized_sentences,
            'sentence_spans':instance.sentence_spans,
            'original_reports':instance.raw_datapoint['reports']
        }


class FullModelInterface(TokenizerInterface):
    def __init__(self):
        super(FullModelInterface, self).__init__()
        self.models = ["code_supervision_only_description", "code_supervision_only_description_unfrozen"]
        self.dps = {k:DefaultProcessor(k) for k in self.models}

    def takes_nl_queries(self):
        return self.dp.takes_nl_queries()

    def get_models(self):
        return self.models

    def query_reports(self, reports, query, is_nl=False, model=None):
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
            'tokenized_text':results['tokenized_text'],
            'sentence_spans':results['sentence_spans'],
            'original_reports':results['original_reports']
        }
        if 'scores' in results.keys():
            return_dict['score'] = results['scores'].item()
        return return_dict
