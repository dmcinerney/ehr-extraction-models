pretrained_model = '/home/jered/Documents/projects/clinical-bert/clinical-bert-weights'
code_graph_file = '/home/jered/Documents/data/icd_codes/code_graph_radiology_expanded.pkl'
data_dir = '/home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded'
#data_dir = '/home/jered/Documents/data/Dataset_10-11-2019/preprocessed/mini'
#data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded'
#data_dir = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/mini'

# training params
batch_size = 8
epochs = 2
limit_rows_train = 10000
limit_rows_val = 1000
subbatches = 4
num_workers = 0
checkpoint_every = 10
copy_checkpoint_every = 300
val_every = 10
email_every = 30

#  model params
sentences_per_checkpoint = 13

# email params
smtp_server = 'smtp.gmail.com'
port = 465
sender_email = 'jeredspython@gmail.com'
receiver_email = 'jered.mcinerney@gmail.com'
