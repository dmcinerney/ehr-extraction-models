import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import ClinicalBertSentences
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad


class Model(nn.Module):
    def __init__(self, num_codes, outdim=64, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True):
        super(Model, self).__init__()
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, conditioned_pool=False, truncate_tokens=30, truncate_sentences=100, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            set_dropout(self.clinical_bert_sentences, 0)
            set_require_grad(self.clinical_bert_sentences, False)
        self.code_embeddings = nn.Embedding(num_codes, outdim)
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.linear = nn.Linear(outdim, 1)
        self.device1 = device1
        self.device2 = device2

    def correct_devices(self):
        self.clinical_bert_sentences.correct_devices()
        self.code_embeddings.to(self.device2)
        self.attention.to(self.device2)
        self.linear.to(self.device2)

    def forward(self, article_sentences, article_sentences_lengths, codes, num_codes):
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        article_sentences_lengths, codes, num_codes, encodings, self_attentions, word_level_attentions =\
            article_sentences_lengths.to(self.device2), codes.to(self.device2), num_codes.to(self.device2), encodings.to(self.device2), self_attentions.to(self.device2), word_level_attentions.to(self.device2)
        # b, ns, nt = word_level_attentions.shape
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(
            self_attentions.mean(3).view(b*ns, nl, nt, nt),
            attention_vecs=word_level_attentions.view(b*ns, 1, nt))\
            .view(b, ns, nt)
        code_embeddings = self.code_embeddings(codes)
        key_padding_mask = (article_sentences_lengths == 0)[:,:encodings.size(1)]
        encoding, sentence_level_attentions = self.attention(code_embeddings.transpose(0, 1), encodings.transpose(0, 1), encodings.transpose(0, 1), key_padding_mask=key_padding_mask)
        nq, _, emb_dim = encoding.shape
        word_level_attentions = word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        traceback_word_level_attentions = traceback_word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions*sentence_level_attentions.unsqueeze(3)
        scores = self.linear(encoding)
        return dict(
            scores=scores.transpose(0, 1).squeeze(2),
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths)

def loss_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels, attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    b, nq, ns, nt = attention.shape
    positive_labels = labels.sum()
    negative_labels = num_codes.sum() - positive_labels
    pos_weight = negative_labels/positive_labels
    losses = F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight, reduction='none')
    code_mask = (torch.arange(nq, device=labels.device) < num_codes.unsqueeze(1))
    loss = losses[code_mask].mean()*b
    if attention_sparsity:
        loss += gamma*entropy(attention.view(b, nq, ns*nt))[code_mask].mean()*b
    if traceback_attention_sparsity:
        loss += gamma*entropy(traceback_attention.view(b, nq, ns*nt))[code_mask].mean()*b
    return loss

def loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    def loss_func_wrapper(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels):
        return loss_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels,
                         attention_sparsity=attention_sparsity, traceback_attention_sparsity=traceback_attention_sparsity, gamma=gamma)
    return loss_func_wrapper

def statistics_func(scores, num_codes, attention, traceback_attention, article_sentences_lengths, labels):
    code_mask = (torch.arange(labels.size(1), device=labels.device) < num_codes.unsqueeze(1))
    b, nq, ns, nt = attention.shape
    return {'positives':(scores[code_mask] > 0).sum(),
            'true_positives':((scores[code_mask] > 0) & (labels[code_mask] == 1)).sum(),
            'relevants':labels[code_mask].sum(),
            'total_predicted':code_mask.sum(),
            'accuracy_sum':((scores[code_mask] > 0).long() == labels[code_mask]).sum().float()*b/code_mask.sum(),
            'attention_entropy':entropy(attention.view(b, nq, ns*nt))[code_mask].mean()*b,
            'traceback_attention_entropy':entropy(traceback_attention.view(b, nq, ns*nt))[code_mask].mean()*b}
