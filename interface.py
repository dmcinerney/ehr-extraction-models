from dataset_scripts.ehr.datapoint_processor import DefaultProcessor

dp = DefaultProcessor()

def get_queries():
    return dp.batcher.codes

def query_text(text, query):
    results, _ = dp.process_datapoint(text, query)
    return {
        'score':results['scores'].item(),
        'heatmaps':{
            'attention':[sent[:len(results['tokenized_text'][i])]
                for i,sent in enumerate(results['attention'][0,0].tolist())],
            'traceback_attention':[sent[:len(results['tokenized_text'][i])]
                for i,sent in enumerate(results['traceback_attention'][0,0].tolist())],
        },
        'tokenized_text':results['tokenized_text']
    }
