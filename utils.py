from contextlib import contextmanager
import sys
import os
import torch

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

def get_valid_queries(file):
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
