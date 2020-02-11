pretrained_model = '/home/jered/Documents/projects/clinical-bert/clinical-bert-weights'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl'
data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded'

# training params
batch_size = 8
epochs = 2
limit_rows_train = None
limit_rows_val = None
subbatches = 4
num_workers = 4
checkpoint_every = 10
val_every = 10
