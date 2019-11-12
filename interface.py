from dataset_scripts.ehr.code_dataset.datapoint_processor import DefaultProcessor
from dataset_scripts.ehr.code_dataset.batcher import Batcher
from pytt.utils import read_pickle

codes_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology.pkl'
model_file = '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint/model_state.tpkl'

class FullModelInterface:
    def __init__(self):
        self.dp = DefaultProcessor(codes_file, model_file, 'code_supervision_individual_sentence')
        #model_file = '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/checkpoint4/model_state.tpkl'
        #self.dp = DefaultProcessor(codes_file, model_file, 'code_supervision')

    def get_queries(self):
        return {k:self.dp.batcher.code_graph.nodes[k]['description']
                if 'description' in self.dp.batcher.code_graph.nodes[k].keys() else ''
                for k,v in self.dp.batcher.code_idxs.items()}

    def query_text(self, text, query):
        results = self.dp.process_datapoint(text, query).results
        attention = results['attention'][0,0]
        attention = attention/attention.max()
        traceback_attention = results['traceback_attention'][0,0]
        traceback_attention = traceback_attention/traceback_attention.max()
        return {
            'score':results['scores'].item(),
            'heatmaps':{
                'attention':[sent[:len(results['tokenized_text'][i])]
                    for i,sent in enumerate(attention.tolist())],
                'traceback_attention':[sent[:len(results['tokenized_text'][i])]
                    for i,sent in enumerate(traceback_attention.tolist())],
            },
            'tokenized_text':results['tokenized_text']
        }

class TokenizerInterface:
    def __init__(self):
        self.batcher = Batcher(read_pickle(codes_file))

    def get_queries(self):
        return {k:self.batcher.code_graph.nodes[k]['description']
                if 'description' in self.batcher.code_graph.nodes[k].keys() else ''
                for k,v in self.batcher.code_idxs.items()}

    def tokenize(self, text):
        return self.batcher.process_datapoint({'reports':text, 'targets':[]}).tokenized_sentences
