pretrained_model = '/home/jered/Documents/projects/clinical-bert/clinical-bert-weights'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl'
data_dir = '/home/jered/Documents/data/Dataset_10-11-2019/FinalPreprocessedData/reports_and_codes_expanded'

# training params
batch_size = 8
epochs = 1
limit_rows_train = None
limit_rows_val = None
subbatches = 4
num_workers = 0
checkpoint_every = 10
val_every = 10

#  model_params
sentences_per_checkpoint = 12
