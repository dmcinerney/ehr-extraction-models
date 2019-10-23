import os
import torch
import pandas as pd
from pytt.utils import seed_state, set_random_state, read_pickle
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.distributed import distributed_wrapper
from pytt.testing.tester import Tester
from pytt.logger import logger
from dataset import Dataset
from dataset_scripts.ehr.summary_dataset.batcher import UnsupervisedBatcher
from models.ehr_extraction.summary_supervision.model import Model, loss_func, statistics_func
from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
import json

val_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_to_seq/val.data'
load_checkpoint_folder = 'checkpoints/ehr_extraction_summary_supervision/checkpoint2'
#load_checkpoint_folder = None
results_directory = 'checkpoints/ehr_extraction_summary_supervision/checkpoint2/results'

class TestFunc:
    def __init__(self, results_directory, tokenizer):
        self.results_directory = results_directory
        self.tokenizer = tokenizer

    def __call__(self, batch, outputs):
        kwargs = {k:outputs[k] for k in
                  ['instance_losses', 'sentence_level_attentions', 'decoded_summary_length', 'attention', 'traceback_attention', 'article_sentences_lengths']}
        stats = {'loss':loss_func(**kwargs), **statistics_func(**kwargs)}
        for i,decoded_summary in enumerate(outputs['decoded_summary']):
#            with open(os.path.join(self.results_directory, 'decoded_summary%i.txt' % batch.instances[i]['df_index']), 'w') as f:
#                decoded_summary_text = self.tokenizer.decode(decoded_summary.tolist())
#                decoded_summary_text = decoded_summary_text if isinstance(decoded_summary_text, str) else decoded_summary_text[0]
#                f.write(decoded_summary_text)
#            with open(os.path.join(self.results_directory, 'reference_summary%i.txt' % batch.instances[i]['df_index']), 'w') as f:
#                tokenized_summary_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(batch.instances[i]['tokenized_summary']))
#                tokenized_summary_text = tokenized_summary_text if isinstance(tokenized_summary_text, str) else tokenized_summary_text[0]
#                f.write(tokenized_summary_text[0])
            with open(os.path.join(self.results_directory, 'attn_vis_data%i.json' % batch.instances[i]['df_index']), 'w') as f:
                vis = {}
                import pdb; pdb.set_trace()
                vis['article_lst'] = sum(batch.instances[i]['tokenized_sentences'], [])
                vis['decoded_lst'] = self.tokenizer.convert_ids_to_tokens(decoded_summary.tolist())
                vis['abstract_str'] = str(' '.join(batch.instances[i]['tokenized_summary']))
                vis['attn_dists'] = outputs['traceback_attention'][i].tolist()
                vis['p_gens'] = [0 for _ in vis['decoded_lst']]
                json.dump(vis, f)
        return None, stats

def main(load_checkpoint_folder=None):
    if load_checkpoint_folder is None:
        seed_state()
    else:
        set_random_state(read_pickle(os.path.join(load_checkpoint_folder, 'random_state.pkl')))
    logger.set_verbosity(2)
#    device1 = 'cuda:%i' % (torch.distributed.get_rank() if torch.distributed.is_initialized() else 1)
    device1 = 'cpu'
    device2 = 'cpu'
    df = pd.read_csv(val_file, compression='gzip')
    val_dataset = Dataset(df[df.impression == df.impression])
    batcher = UnsupervisedBatcher()
    batch_size = 4
    val_indices_iterator = init_indices_iterator(len(val_dataset), batch_size)
    val_iterator = batcher.batch_iterator(val_dataset, val_indices_iterator, subbatches=4)
    start_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.cls_token])[0]
    stop_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.sep_token])[0]
    mask_token_id = batcher.tokenizer.convert_tokens_to_ids([batcher.tokenizer.mask_token])[0]
    model = Model({'start':start_token_id, 'stop':stop_token_id, 'mask':mask_token_id}, batcher.tokenizer.vocab_size, sentences_per_checkpoint=5, device1=device1, device2=device2)
    if load_checkpoint_folder is not None:
        model.load_state_dict(torch.load(os.path.join(load_checkpoint_folder, 'model_state.tpkl'), map_location='cpu'))
    model.correct_devices()
    if torch.distributed.is_initialized():
        model = LDDP(model, torch.distributed.get_world_size())
    model.eval()
    tester = Tester(model, val_iterator)
#    test_func = TestFunc(results_directory, batcher.tokenizer, tensorboard_dir=os.path.join(save_checkpoint_folder, 'tensorboard/with_dropout'))
    test_func = TestFunc(results_directory, batcher.tokenizer, tensorboard_dir=os.path.join(save_checkpoint_folder, 'tensorboard/without_dropout'))
    tester.test(test_func)

if __name__ == '__main__':
    main(load_checkpoint_folder=load_checkpoint_folder)
#    nprocs = 2
#    main_distributed = distributed_wrapper(main, nprocs)
#    main_distributed(load_checkpoint_folder=load_checkpoint_folder)
