import os
import torch
from pytt.batching.postprocessor import StandardPostprocessor
from pytt.utils import read_pickle
from utils import entropy
import pandas as pd

class Postprocessor(StandardPostprocessor):
    def __init__(self, graph_ops, code_idxs, output_batch_class):
        self.graph_ops = graph_ops
        self.code_names = {v:k for k,v in code_idxs.items()}
        self.output_batch_class = output_batch_class
        self.dir = None

    def add_output_dir(self, dir):
        self.dir = dir
        #self.code_counts = read_pickle(os.path.join(dir, 'counts.pkl'))
        if os.path.exists(os.path.join(dir, 'summary_stats.csv')):
            raise Exception

    def output_batch(self, batch, outputs):
        outputs['total_num_codes'] = torch.tensor(len(self.code_names))
        return self.output_batch_class.from_outputs(self, batch, outputs)

    def add_summary_stats(self, batch, outputs):
        if self.dir is None:
            print("No output directory! Results not being recorded.")
            return
        b, nq, ns, nt = outputs['attention'].shape
        attention_entropy = entropy(outputs['attention'].view(b, nq, ns*nt))
        traceback_attention_entropy = entropy(outputs['traceback_attention'].view(b, nq, ns*nt))
        columns = ['code_name', 'code_idx', 'attention', 'traceback_attention', 'label', 'score', 'depth', 'num_report_sentences']
        rows = []
        for b in range(len(batch)):
            for s in range(outputs['num_codes'][b]):
                code = outputs['codes'][b, s].item()
                codename = self.code_names[code]
                attention = attention_entropy[b, s].item()
                traceback_attention = traceback_attention_entropy[b, s].item()
                label = outputs['labels'][b, s].item()
                score = outputs['scores'][b, s].item()
                depth = self.graph_ops.depth(codename)
                num_report_sentences = (outputs['article_sentences_lengths'][b] > 0).sum()
                rows.append([
                    codename,
                    code,
                    attention,
                    traceback_attention,
                    label,
                    score,
                    depth,
                    num_report_sentences,
                ])
        df = pd.DataFrame(rows, columns=columns)
        file = os.path.join(self.dir, 'summary_stats.csv')
        header = True if not os.path.exists(file) else False
        df.to_csv(file, mode='a', header=header)
