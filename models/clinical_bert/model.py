import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from utils import directory
from transformers import BertModel
import models.clinical_bert.parameters as p

class ClinicalBertWrapper(nn.Module):
    def __init__(self):
        super(ClinicalBertWrapper, self).__init__()
        config = BertModel.config_class.from_pretrained(p.pretrained_model)
        config.output_attentions = True
        self.hidden_size = config.hidden_size
        self.clinical_bert = BertModel.from_pretrained(p.pretrained_model, config=config)

    def forward(self, token_ids, attention_mask):
        outputs = self.clinical_bert(token_ids, attention_mask=attention_mask)
        encodings = outputs[0]
        self_attentions = torch.cat([self_attention.unsqueeze(1) for self_attention in outputs[2]], 1)
        return encodings, self_attentions

class ClinicalBertSentences(nn.Module):
    def __init__(self, embedding_dim=None, conditioned_pool=False, truncate_sentences=None, truncate_tokens=None, sentences_per_checkpoint=10, device='cpu'):
        super(ClinicalBertSentences, self).__init__()
        self.clinical_bert_wrapper = ClinicalBertWrapper()
        self.device = device
        outdim = self.clinical_bert_wrapper.hidden_size
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            self.linear = nn.Linear(outdim, embedding_dim)
        self.conditioned_pool = conditioned_pool
        if self.conditioned_pool:
            self.attention = nn.MultiheadAttention(embedding_dim, 1)
        self.truncate_sentences = truncate_sentences
        self.truncate_tokens = truncate_tokens
        self.sentences_per_checkpoint = sentences_per_checkpoint

    def correct_devices(self):
        self.to(self.device)

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
                self.run_checkpointed_clinical_bert, article_sentences_temp.reshape(b*actual_ns, nt), mask_temp.reshape(b*actual_ns, nt), conditioning_reshaped, *self.parameters())
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
        token_ids, attention_mask, conditioning = token_ids.to(self.device), attention_mask.to(self.device), conditioning.to(self.device)
        encodings, self_attentions = self.clinical_bert_wrapper(token_ids, attention_mask)
        if self.conditioned_pool and conditioning.size(0) != 0:
            if self.embedding_dim is not None:
                encodings = self.linear(encodings)
            encodings, word_level_attentions = self.attention(conditioning.unsqueeze(0), encodings.transpose(0, 1), encodings.transpose(0, 1))
            encodings = encodings.squeeze(0)
            word_level_attentions.squeeze(1)
            return encodings, self_attentions, word_level_attentions
        else:
            if conditioning.size(0) != 0:
                raise Exception
            encodings = encodings[:,0]
            if self.embedding_dim is not None:
                encodings = self.linear(encodings)
            # need to push making word_level_attentions (1 on the first element of the sentences)
            # until after checkpoint because it won't have a gradient
            return encodings, self_attentions
