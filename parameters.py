pretrained_model = '/home/jered/Documents/projects/clinical-bert/clinical-bert-weights'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl'
#data_dir = '/home/jered/Documents/data/Dataset_10-11-2019/FinalPreprocessedData/reports_and_codes_expanded'
#data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded'
data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/mini'

# training params
batch_size = 8
epochs = 4
limit_rows_train = None
limit_rows_val = None
subbatches = 4
num_workers = 0
checkpoint_every = 10
copy_checkpoint_every = 100
val_every = 10
email_every = 30
sender_email = 'jeredspython@gmail.com'
receiver_email = 'jered.mcinerney@gmail.com'

#  model_params
sentences_per_checkpoint = 17
