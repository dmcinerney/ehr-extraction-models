import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
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

class ClinicalBertWrapper(nn.Module):
    def __init__(self):
        super(ClinicalBertWrapper, self).__init__()
        self.clinical_bert = BertModel.from_pretrained(p.pretrained_model)
        self.attention_tracker = AttentionTracker()
        for name,module in self.clinical_bert.named_modules():
            if isinstance(module, BertSelfAttention):
                advanced_setattr(self.clinical_bert, name, BertSelfAttentionwithTracking(module, self.attention_tracker))

    def forward(self, token_ids, attention_mask):
        outputs = self.clinical_bert(token_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        self_attentions = self.attention_tracker.get_attention() # batch_size x num_layers x num_heads x num_queries x num_keys
        self.attention_tracker.clear()
        return outputs, self_attentions

def advanced_setattr(obj, name, value):
    parts = name.split('.')
    if len(parts) > 1:
        advanced_setattr(getattr(obj, parts[0]), '.'.join(parts[1:]), value)
    else:
        setattr(obj, name, value)

class ClinicalBertSentences(nn.Module):
    def __init__(self, embedding_dim=None, conditioned_pool=False, truncate_sentences=None, truncate_tokens=None, sentences_per_checkpoint=10):
        super(ClinicalBertSentences, self).__init__()
        outdim = 768
        self.clinical_bert_wrapper = ClinicalBertWrapper()
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            self.linear = nn.Linear(outdim, embedding_dim)
        self.conditioned_pool = conditioned_pool
        if self.conditioned_pool:
            self.attention = nn.MultiheadAttention(embedding_dim, 1)
        self.truncate_sentences = truncate_sentences
        self.truncate_tokens = truncate_tokens
        self.sentences_per_checkpoint = sentences_per_checkpoint

    def forward(self, article_sentences, article_sentences_lengths, conditioning=None):
        b, ns, nt = article_sentences.shape
        mask = torch.arange(nt, device=article_sentences.device).view(1,1,-1) < article_sentences_lengths.unsqueeze(2)
        if self.truncate_sentences is not None:
            ns = min(self.truncate_sentences, ns)
        if self.truncate_tokens is not None:
            nt = min(self.truncate_tokens, nt)
        ns_temp = self.sentences_per_checkpoint
        article_sentences = article_sentences[:,:ns,:nt]
        mask = mask[:,:ns,:nt]
        encodings, self_attentions, word_level_attentions = [], [], []
        if conditioning is None:
            conditioning = torch.zeros(0)
        for offset in range(0, ns, ns_temp):
            article_sentences_temp, mask_temp = article_sentences[:,offset:offset+ns_temp], mask[:,offset:offset+ns_temp]
            actual_ns = mask_temp.size(1)
            if conditioning.size(0) != 0:
                conditioning_reshaped = conditioning.view(b, 1, conditioning.size(1))\
                                                    .expand(b, actual_ns, conditioning.size(1))\
                                                    .view(b*actual_ns, conditioning.size(1))
            else:
                conditioning_reshaped = conditioning
            results = checkpoint(
                self.run_checkpointed_clinical_bert, article_sentences_temp.view(b*actual_ns, nt), mask_temp.view(b*actual_ns, nt), conditioning_reshaped, *self.parameters())
            if len(results) == 3:
                encodings_temp, self_attentions_temp, word_level_attentions_temp = results
            else:
                encodings_temp, self_attentions_temp = results
                word_level_attentions_temp = torch.zeros((b*actual_ns, nt), device=encodings_temp.device)
                word_level_attentions_temp[:, 0] = 1
            encodings.append(encodings_temp.view(b, actual_ns, -1))
            nl, nh = self_attentions_temp.shape[1:3]
            self_attentions.append(self_attentions_temp.view(b, actual_ns, nl, nh, nt, nt))
            word_level_attentions.append(word_level_attentions_temp.view(b, actual_ns, nt))
        encodings = torch.cat(encodings, 1)
        self_attentions = torch.cat(self_attentions, 1)
        word_level_attentions = torch.cat(word_level_attentions, 1)
        return encodings, self_attentions, word_level_attentions

    def run_checkpointed_clinical_bert(self, token_ids, attention_mask, conditioning, *args):
        encodings, self_attentions = self.clinical_bert_wrapper(token_ids, attention_mask)
        if self.conditioned_pool and conditioning.size(0) != 0:
            encodings = encodings[0]
            if self.embedding_dim is not None:
                encodings = self.linear(encodings)
            encodings, word_level_attentions = self.attention(conditioning.unsqueeze(0), encodings.transpose(0, 1), encodings.transpose(0, 1))
            encodings = encodings.squeeze(0)
            word_level_attentions.squeeze(1)
            return encodings, self_attentions, word_level_attentions
        else:
            if conditioning.size(0) != 0:
                raise Exception
            encodings = encodings[1]
            if self.embedding_dim is not None:
                encodings = self.linear(encodings)
            # need to push making word_level_attentions (1 on the first element of the sentences)
            # until after checkpoint because it won't have a gradient
            return encodings, self_attentions
