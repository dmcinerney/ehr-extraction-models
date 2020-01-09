import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.clinical_bert.model import ClinicalBertSentences
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts, none_to_tensor, tensor_to_none


class Model(nn.Module):
    def __init__(self, num_codes, outdim=64, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True, reduce_code_embeddings=False, dropout=.15):
        super(Model, self).__init__()
        self.num_codes = num_codes
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            self.freeze_bert()
        else:
            self.unfreeze_bert(dropout=dropout)
        self.code_embeddings = nn.Embedding(num_codes, outdim)
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.linear = nn.Linear(outdim, 1)
        self.linear2 = nn.Linear(outdim*2, outdim) if reduce_code_embeddings else None
        self.device1 = device1
        self.device2 = device2

    def freeze_bert(self):
        set_dropout(self.clinical_bert_sentences, 0)
        set_require_grad(self.clinical_bert_sentences, False)

    def unfreeze_bert(self, dropout=.15):
        set_dropout(self.clinical_bert_sentences, dropout)
        set_require_grad(self.clinical_bert_sentences, True)

    def correct_devices(self):
        self.clinical_bert_sentences.correct_devices()
        self.code_embeddings.to(self.device2)
        self.attention.to(self.device2)
        self.linear.to(self.device2)
        if self.linear2 is not None:
            self.linear2.to(self.device2)
        self.codes_per_checkpoint = 1000

    def forward(self, article_sentences, article_sentences_lengths, num_codes, codes=None, code_description=None, code_description_length=None):
        nq = num_codes.max()
        nq_temp =  self.codes_per_checkpoint
        scores, attention, traceback_attention = [], [], []
        for offset in range(0, nq, nq_temp):
            if codes is not None:
                codes_temp = codes[:, offset:offset+nq_temp]
            else:
                codes_temp = torch.zeros(0)
            if code_description is not None:
                code_description_temp = code_description[:, offset:offset+nq_temp]
                code_description_length_temp = code_description_length[:, offset:offset+nq_temp]
            else:
                code_description_temp = torch.zeros(0)
                code_description_length_temp = torch.zeros(0)
            num_codes_temp = torch.clamp(num_codes-offset, 0, nq_temp)
            scores_temp, attention_temp, traceback_attention_temp = checkpoint(
                self.inner_forward,
                article_sentences,
                article_sentences_lengths,
                num_codes_temp,
                codes_temp,
                code_description_temp,
                code_description_length_temp,
                *self.parameters())
            scores.append(scores_temp)
            attention.append(attention_temp)
            traceback_attention.append(traceback_attention_temp)
        scores = torch.cat(scores, 1)
        attention = torch.cat(attention, 1)
        traceback_attention = torch.cat(traceback_attention, 1)
        return_dict = dict(
            scores=scores,
            num_codes=num_codes,
            total_num_codes=torch.tensor(self.num_codes),
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths)
        if codes is not None:
            return_dict['codes'] = codes
        return return_dict

    def inner_forward(self, article_sentences, article_sentences_lengths, num_codes, codes, code_description, code_description_length, *args):
        codes, code_description, code_description_length = tensor_to_none(codes), tensor_to_none(code_description), tensor_to_none(code_description_length)
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        article_sentences_lengths, num_codes, encodings, self_attentions, word_level_attentions =\
            article_sentences_lengths.to(self.device2), num_codes.to(self.device2), encodings.to(self.device2), self_attentions.to(self.device2), word_level_attentions.to(self.device2)
        if codes is not None:
            codes = codes.to(self.device2)
        # b, ns, nt = word_level_attentions.shape
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(
            self_attentions.mean(3).view(b*ns, nl, nt, nt),
            attention_vecs=word_level_attentions.view(b*ns, 1, nt))\
            .view(b, ns, nt)
        if codes is None and code_description is None: raise Exception
        if codes is not None:
            code_embeddings = self.code_embeddings(codes)
        if code_description is not None:
            code_description_embeddings = self.clinical_bert_sentences(
                code_description,
                code_description_length,
            )[0].to(self.device2)
        if codes is not None and code_description is not None:
            code_embeddings = torch.cat((code_embeddings, code_description_embeddings), 2)
            code_embeddings = self.linear2(code_embeddings)
        elif codes is None:
            code_embeddings = code_description_embeddings
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
        return scores.transpose(0, 1).squeeze(2), attention, traceback_attention

def loss_func(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels, attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
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
    def loss_func_wrapper(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels):
        return loss_func(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels,
                         attention_sparsity=attention_sparsity, traceback_attention_sparsity=traceback_attention_sparsity, gamma=gamma)
    return loss_func_wrapper

def statistics_func(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels):
    b, nq, ns, nt = attention.shape
    code_mask = (torch.arange(labels.size(1), device=labels.device) < num_codes.unsqueeze(1))
    positives = get_code_counts(total_num_codes, codes, code_mask, (scores > 0))
    true_positives = get_code_counts(total_num_codes, codes, code_mask, ((scores > 0) & (labels == 1)))
    relevants = get_code_counts(total_num_codes, codes, code_mask, labels)
    return {'positives':positives,
            'true_positives':true_positives,
            'relevants':relevants,
            'total_predicted':code_mask.sum(),
            'accuracy_sum':((scores[code_mask] > 0).long() == labels[code_mask]).sum().float()*b/code_mask.sum(),
            'attention_entropy':entropy(attention.view(b, nq, ns*nt))[code_mask].mean()*b,
            'traceback_attention_entropy':entropy(traceback_attention.view(b, nq, ns*nt))[code_mask].mean()*b}
