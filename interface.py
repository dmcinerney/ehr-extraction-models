from dataset_scripts.ehr.datapoint_processor import DefaultProcessor

codes_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes/codes.pkl'
model_file = '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/clinical_bert_mimic_extraction/checkpoint/model_state.tpkl'
dp = DefaultProcessor(codes_file, model_file)

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
