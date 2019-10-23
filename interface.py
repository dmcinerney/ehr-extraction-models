from dataset_scripts.ehr.code_dataset.datapoint_processor import DefaultProcessor

codes_file = '/home/jered/Documents/data/icd_codes/code_graph.pkl'
model_file = '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/checkpoint6/model_state.tpkl'
dp = DefaultProcessor(codes_file, model_file)

def get_queries():
    return dp.batcher.code_idxs

def query_text(text, query):
    results = dp.process_datapoint(text, query).results
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
