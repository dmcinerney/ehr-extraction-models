import os
import torch
from pytt.batching.postprocessor import StandardPostprocessor
from pytt.utils import read_pickle
from utils import entropy
import pandas as pd

class Postprocessor(StandardPostprocessor):
    def __init__(self, hierarchy, code_idxs, output_batch_class):
        self.hierarchy = hierarchy
        self.code_names = {v:k for k,v in code_idxs.items()}
        self.output_batch_class = output_batch_class
        self.dir = None
        self.k = 100

    def add_output_dir(self, dir):
        self.dir = dir
        #self.code_counts = read_pickle(os.path.join(dir, 'counts.pkl'))
        if os.path.exists(os.path.join(dir, 'summary_stats.csv')):
            raise Exception
        self.summaries_dir = os.path.join(self.dir, 'summaries')
        os.mkdir(self.summaries_dir)
        self.system_dir = os.path.join(self.summaries_dir, 'system')
        os.mkdir(self.system_dir)
        self.reference_dir = os.path.join(self.summaries_dir, 'reference')
        os.mkdir(self.reference_dir)

    def output_batch(self, batch, outputs):
        outputs['total_num_codes'] = torch.tensor(len(self.code_names))
        return self.output_batch_class.from_outputs(self, batch, outputs)

    def add_summary_stats(self, batch, outputs):
        if self.dir is None:
            print("No output directory! Results not being recorded.")
            return
        supervised = 'annotations' in batch.instances[0].keys()
        b, nq, ns, nt = outputs['attention'].shape
        attention_entropy = entropy(outputs['attention'].view(b, nq, ns*nt))
        traceback_attention_entropy = entropy(outputs['traceback_attention'].view(b, nq, ns*nt))
        columns = ['code_name', 'code_idx', 'attention_entropy', 'traceback_attention_entropy', 'label', 'score', 'depth',
                   'num_report_sentences', 'num_report_clusters', 'patient_id', 'timepoint_id',
#                   'reference_sentence_indices', 'reference_sentence_rankings', 'reference_sentence_attention']
                   'reference_sentence_indices', 'reference_sentence_rankings', 'sentence_attention']
        rows = []
        for b in range(len(batch)):
            patient_id = int(batch.instances[b]['original_reports'].patient_id.iloc[0])
            last_report_id = batch.instances[b]['original_reports'].index[-1]
            tokenized_sentences = batch.instances[b]['tokenized_sentences']
            if supervised:
                annotations = eval(batch.instances[b]['annotations'])
            for s in range(outputs['num_codes'][b]):
                code = outputs['codes'][b, s].item()
                codename = self.code_names[code]
                attn_ent = attention_entropy[b, s].item()
                traceback_attn_ent = traceback_attention_entropy[b, s].item()
                label = outputs['labels'][b, s].item()
                score = outputs['scores'][b, s].item() if 'scores' in outputs.keys() else None
                depth = self.hierarchy.depth(codename)
                num_report_sentences = (outputs['article_sentences_lengths'][b] > 0).sum().item()
                if supervised:
                    num_report_clusters = len(outputs['clustering'][b][s])
                    sentences = [' '.join(tokenized_sentences[cluster[0]]) for cluster in outputs['clustering'][b][s]]
                    summary = '\n'.join(sentences[:self.k])
                    id = len(os.listdir(self.system_dir))
                    with open(os.path.join(self.system_dir, 'summary_%i_system.txt' % id), 'w') as f:
                        f.write(summary)
                    reference_sentence_indices_set = set([])
                    for annotator,v in annotations.items():
                        reference_sentence_indices = [int(i) for i in v['past-reports']['tag_sentences'][codename]]
                        reference_sentence_indices_set.update(reference_sentence_indices)
                        reference = '\n'.join([' '.join(tokenized_sentences[i]) for i in reference_sentence_indices])
                        with open(os.path.join(self.reference_dir, 'summary_%i_%s.txt' % (id, annotator)), 'w') as f:
                            f.write(reference)
                    sentence_to_ranking = {sentence_idx:i for i in range(len(sentences)) for sentence_idx in outputs['clustering'][b][s][i]}
                    reference_sentence_indices = sorted(list(reference_sentence_indices_set))
                    reference_sentence_rankings = [sentence_to_ranking[i] for i in sorted(list(reference_sentence_indices_set))]
#                    reference_sentence_attention = [outputs['attention'][b, s, i].sum().item() for i in sorted(list(reference_sentence_indices_set))]
                    sentence_attention = [outputs['attention'][b, s, outputs['clustering'][b][s][i][0]].sum().item() for i in range(len(sentences))]
                else:
                    num_report_clusters = None
                    reference_sentence_indices = None
                    reference_sentence_rankings = None
#                    reference_sentence_attention = None
                    sentence_attention = None
                # NOTE: cannot include summaries here because this file might be emailed and summaries contain phi!
                rows.append([
                    codename,
                    code,
                    attn_ent,
                    traceback_attn_ent,
                    label,
                    score,
                    depth,
                    num_report_sentences,
                    num_report_clusters,
                    patient_id,
                    last_report_id,
                    reference_sentence_indices,
                    reference_sentence_rankings,
#                    reference_sentence_attention,
                    sentence_attention,
                ])
        df = pd.DataFrame(rows, columns=columns)
        file = os.path.join(self.dir, 'summary_stats.csv')
        header = True if not os.path.exists(file) else False
        df.to_csv(file, mode='a', header=header)

    def get_summary_attachment_generator(self):
        with open(os.path.join(self.dir, 'summary_stats.csv'), 'rb') as file:
            yield 'summmary_stats', 'summary_stats.csv', file
