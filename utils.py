from contextlib import contextmanager
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

@contextmanager
def directory(new_dir):
    path = sys.path[0]
    current_path_files = [os.path.join(path,file) for file in os.listdir()]
    names = [name for name,module in sys.modules.items() if hasattr(module, '__file__') and module.__file__ in current_path_files]
    newnames = [path+name for name in names]
    replace_keys(sys.modules, names, newnames)

    sys.path.insert(0, new_dir)

    original_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(original_dir)

    del sys.path[0]

    replace_keys(sys.modules, newnames, names)


def replace_keys(dictionary, old_keys, new_keys):
    for old_key,new_key in zip(old_keys, new_keys):
        dictionary[new_key] = dictionary[old_key]
        del dictionary[old_key]

def traceback_attention(self_attentions, attention_vecs=None, factor_in_residuals=True):
    # expects a self_attentions matrix of shape:
    #   batch_size x num_layers x num_queries x num_keys
    #   IMPORTANT NOTE: this does not contain a num_heads dim
    #     because this should be averaged out
    # expects an attention_vecs matrix of shape:
    #   batch_size x num_vecs x num_queries
    # returns an attention_vecs matrix of shape:
    #   batch_size x num_vecs x num_queries
    b, num_layers, num_queries, num_keys = self_attentions.size()
    if attention_vecs is None:
        attention_vecs = \
            torch.eye(num_queries, device=self_attentions.device)\
            .expand(b, num_queries, num_queries)
    for i in range(self_attentions.size(1)-1,-1,-1):
        self_attention_matrix = self_attentions[:,i]
        if factor_in_residuals:
            self_attention_matrix = self_attention_matrix\
                + torch.eye(self_attention_matrix.size(1),
                            device=self_attention_matrix.device)
        attention_vecs = attention_vecs @ self_attention_matrix
        if factor_in_residuals:
            attention_vecs = attention_vecs/attention_vecs.sum(2, keepdim=True)
    return attention_vecs

def entropy(attention):
    return torch.distributions.OneHotCategorical(attention).entropy()

def set_dropout(model, p):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p

def set_require_grad(model, boolean):
    for p in model.parameters():
        p.requires_grad = boolean

def get_code_counts(total_num_codes, codes, code_mask, satisfied_condition):
    template_counts = torch.zeros((codes.size(0), total_num_codes), device=codes.device)
    return template_counts.scatter_add(1, codes, satisfied_condition.masked_fill(code_mask==0, 0).float()).sum(0)

def get_queries(file):
    if os.path.exists(file):
        with open(file, 'r') as f:
            return eval(f.read())

def none_to_tensor(v):
    if v is None:
        return torch.zeros(0)
    else:
        return v

def tensor_to_none(t):
    if t.dim() == 1 and t.size(0) == 0:
        return None
    else:
        return t

def precision_recall_f1(true_positives, positives, relevants, reduce=None):
    mask = (positives != 0) | (relevants != 0)
    if reduce == 'micro':
        true_positives, positives, relevants = true_positives.sum(), positives.sum(), relevants.sum()
    precision = (true_positives/positives).masked_fill(positives == 0, 0)
    recall = (true_positives/relevants).masked_fill(relevants == 0, 0)
    f1 = (2*precision*recall/(precision + recall)).masked_fill((precision+recall) == 0, 0)
    if reduce == 'macro':
        precision, recall, f1 = precision[mask].mean(), recall[mask].mean(), f1[mask].mean()
    if reduce is not None:
        precision, recall, f1 = precision.item(), recall.item(), f1.item()
    return precision, recall, f1

def plot_stacked_bar(data, x_ticks=None, stack_labels=None, y_label=None, title=None, show_nums=None, y_lim=None, file=None, figsize=None):
    ind = np.arange(len(data[0][0]))    # the x locations for the groups
    width = 0.40       # the width of the bars: can also be len(x) sequence
    figure = plt.figure(figsize=figsize)
    ps = [plt.bar(
        ind,
        mean,
        width,
        bottom=(data[i-1][0] if i > 0 else None),
        yerr=error
    ) for i,(mean,error) in enumerate(data)]

    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if x_ticks is not None:
        plt.xticks(ind, x_ticks)
    if stack_labels is not None:
        plt.legend(tuple(p[0] for p in ps), stack_labels)
    for i,bar in enumerate(ps):
        for j,patch in enumerate(bar):
            if show_nums is None or show_nums[i,j]:
            # get_width pulls left or right; get_y pushes up or down
                plt.text(patch.get_x(), sum(p[j].get_height() for p in ps[:i+1])+.005, \
                        str(round(sum(mean[j] for mean,_ in data[:i+1]), 4)), fontsize=12)
    if y_lim is not None:
        plt.ylim(y_lim)
    if file is not None:
        plt.savefig(file)
    return figure
