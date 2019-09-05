import math
import torch
from torch import nn
from utils import directory
import models.clinical_bert.parameters as p
with directory(p.path_to_clinical_bert_repo):
    from modeling_readmission import BertModel, BertSelfAttention


class BertSelfAttentionwithTracking(nn.Module):
    def __init__(self, bert_self_attention_module, attention_tracker):
        super(BertSelfAttentionwithTracking, self).__init__()
        self.bert_self_attention_module = bert_self_attention_module
        self.attention_tracker = attention_tracker

    # IMPORTANT NOTE: this is taken from https://github.com/kexinhuang12345/clinicalBERT/blob/master/modeling_readmission.py
    #   so it can be modified to record the attention scores
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.bert_self_attention_module.query(hidden_states)
        mixed_key_layer = self.bert_self_attention_module.key(hidden_states)
        mixed_value_layer = self.bert_self_attention_module.value(hidden_states)

        query_layer = self.bert_self_attention_module.transpose_for_scores(mixed_query_layer)
        key_layer = self.bert_self_attention_module.transpose_for_scores(mixed_key_layer)
        value_layer = self.bert_self_attention_module.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.bert_self_attention_module.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.bert_self_attention_module.dropout(attention_probs)
        self.attention_tracker.log_attention(attention_probs) # ONLY LINE ADDED

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.bert_self_attention_module.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class AttentionTracker():
    def __init__(self):
        self.clear()

    def log_attention(self, attention):
        self.attentions.append(attention.unsqueeze(1))

    def get_attention(self):
        return torch.cat(self.attentions, 1)

    def clear(self):
        self.attentions = []

class ClinicalBertExtractionModel(nn.Module):
    def __init__(self):
        super(ClinicalBertExtractionModel, self).__init__()
        self.clinical_bert = BertModel.from_pretrained(p.pretrained_model)
        self.attention_tracker = AttentionTracker()
        for name,module in self.clinical_bert.named_modules():
            if isinstance(module, BertSelfAttention):
                advanced_setattr(self.clinical_bert, name, BertSelfAttentionwithTracking(module, self.attention_tracker))
        # TODO: add other layers for classification etc.

    def forward(self, input_ids, attention_mask):
        outputs = self.clinical_bert(input_ids, attention_mask=attention_mask)
        self_attentions = self.attention_tracker.get_attention() # batch_size x num_layers x num_heads x num_queries x num_keys
        self.attention_tracker.clear()
        return outputs, self_attentions

def advanced_setattr(obj, name, value):
    parts = name.split('.')
    if len(parts) > 1:
        advanced_setattr(getattr(obj, parts[0]), '.'.join(parts[1:]), value)
    else:
        setattr(obj, name, value)
