import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import EncoderSentences, ClinicalBertWrapper
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts


class Model(nn.Module):
    def __init__(self, num_codes, outdim=64, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True, reduce_code_embeddings=False):
        super(Model, self).__init__()
        self.num_codes = num_codes
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            self.freeze_bert()
        else:
            self.unfreeze_bert()
        self.code_embeddings = nn.Embedding(num_codes, outdim)
        self.linear = nn.Linear(outdim, 1)
        self.linear2 = nn.Linear(outdim*2, outdim) if reduce_code_embeddings else None
        self.predict_targets = PredictTargets(outdim)
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
        self.linear.to(self.device2)
        if self.linear2 is not None:
            self.linear2.to(self.device2)
        self.predict_targets.to(self.device2)

    def forward(self, article_sentences, article_sentences_lengths, num_codes, codes=None, code_description=None, code_description_length=None):
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
        if codes is None and code_description is None: raise Exception
        if codes is not None:
            code_embeddings = self.code_embeddings(codes)
        if code_description is not None:
            code_description_embeddings = self.clinical_bert_sentences(code_description, code_description_length)[0].to(self.device2)
        if codes is not None and code_description is not None:
            code_embeddings = torch.cat((code_embeddings, code_description_embeddings), 2)
            code_embeddings = self.linear2(code_embeddings)
        elif codes is None:
            code_embeddings = code_description_embeddings
        sentence_level_scores = self.predict_targets(code_embeddings.transpose(0, 1), encodings.transpose(0, 1)).transpose(0, 2).transpose(1, 2)
        mask = (article_sentences_lengths != 0)[:,:encodings.size(1)].unsqueeze(1).expand(sentence_level_scores.shape)
#        sentence_level_scores_masked = sentence_level_scores*mask
        sentence_level_scores_masked = torch.zeros_like(sentence_level_scores).masked_scatter(mask, sentence_level_scores[mask])
        scores = sentence_level_scores_masked.sum(2)/mask.sum(2)
        return_dict = dict(
            scores=scores,
            num_codes=num_codes,
            word_level_attentions=word_level_attentions,
            traceback_word_level_attentions=traceback_word_level_attentions,
            sentence_level_scores=sentence_level_scores,
            article_sentences_lengths=article_sentences_lengths)
        if codes is not None:
            return_dict['codes'] = codes
        return return_dict

def loss_func(total_num_codes, code_idxs, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels):
    b, nq = scores.shape
    positive_labels = labels.sum()
    negative_labels = num_codes.sum() - positive_labels
    pos_weight = negative_labels/positive_labels
    losses = F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight, reduction='none')
    code_mask = (torch.arange(nq, device=labels.device) < num_codes.unsqueeze(1))
    return losses[code_mask].mean()*b

def get_sentence_level_attentions(sentence_level_scores, article_sentences_lengths, labels):
    ns = sentence_level_scores.size(2)
    sentence_level_attentions = sentence_level_scores*(labels*2-1).unsqueeze(2)
    sentence_level_attentions[(article_sentences_lengths == 0)[:,:ns].unsqueeze(1).expand(sentence_level_attentions.shape)] = -float('inf')
    sentence_level_attentions = F.softmax(sentence_level_attentions, 2)
    return sentence_level_attentions

def get_full_attention(word_level_attentions, sentence_level_attentions):
    b, ns, nt = word_level_attentions.shape
    nq = sentence_level_attentions.size(1)
    word_level_attentions = word_level_attentions\
        .view(b, 1, ns, nt)\
        .expand(b, nq, ns, nt)
    attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
    return attention

def statistics_func(total_num_codes, code_idxs, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels):
    b, ns, nt = word_level_attentions.shape
    nq = scores.size(1)
    sentence_level_attentions = get_sentence_level_attentions(sentence_level_scores, article_sentences_lengths, labels)
    attention = get_full_attention(word_level_attentions, sentence_level_attentions)
    traceback_attention = get_full_attention(traceback_word_level_attentions, sentence_level_attentions)
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

class PredictTargets(nn.Module):
    def __init__(self, dim):
        super(PredictTargets, self).__init__()
        self.linear1 = nn.Linear(2*dim, dim)
        self.linear2 = nn.Linear(dim, 1)

    def forward(self, targets, embeddings):
        nt, b, vs = targets.shape
        ne = embeddings.size(0)
        vectors = torch.cat((targets.unsqueeze(1).expand(nt, ne, b, vs), embeddings.unsqueeze(0).expand(nt, ne, b, vs)), 3)
        return self.linear2(F.tanh(self.linear1(vectors))).squeeze(3)
