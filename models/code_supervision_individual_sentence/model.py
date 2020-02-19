import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.clinical_bert.model import ClinicalBertWrapper, EncoderSentences
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts, tensor_to_none


class Model(nn.Module):
    def __init__(self, num_codes=0, num_linearization_embeddings=0, outdim=64, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True, num_code_embedding_types=1, dropout=.15):
        super(Model, self).__init__()
        self.num_codes = num_codes
        self.num_linearization_embeddings = num_linearization_embeddings
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            self.freeze_bert()
        else:
            self.unfreeze_bert(dropout=dropout)
        self.code_embeddings = nn.Embedding(num_codes, outdim) if num_codes > 0 else None
        self.linearized_code_transformer = EncoderSentences(lambda : LinearizedCodesTransformer(num_linearization_embeddings), embedding_dim=outdim, truncate_tokens=50,
                                                            truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device2)\
                                           if num_linearization_embeddings > 0 else None
        self.predict_targets = PredictTargets(outdim)
        self.linear2 = nn.Linear(outdim*num_code_embedding_types, outdim) if num_code_embedding_types > 1 else None
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
        if self.code_embeddings is not None:
            self.code_embeddings.to(self.device2)
        self.predict_targets.to(self.device2)
        if self.linear2 is not None:
            self.linear2.to(self.device2)
        self.codes_per_checkpoint = 1000


    def forward(self, article_sentences, article_sentences_lengths, num_codes, codes=None, code_description=None, code_description_length=None, linearized_codes=None, linearized_codes_lengths=None, linearized_descriptions=None, linearized_descriptions_lengths=None):
        nq = num_codes.max()
        nq_temp =  self.codes_per_checkpoint
        scores, word_level_attentions, traceback_word_level_attentions, sentence_level_scores = [], [], [], []
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
            if linearized_codes is not None:
                linearized_codes_temp = linearized_codes[:, offset:offset+nq_temp]
                linearized_codes_lengths_temp = linearized_codes_lengths[:, offset:offset+nq_temp]
            else:
                linearized_codes_temp = torch.zeros(0)
                linearized_codes_lengths_temp = torch.zeros(0)
            num_codes_temp = torch.clamp(num_codes-offset, 0, nq_temp)
            scores_temp, word_level_attentions_temp, traceback_word_level_attentions_temp, sentence_level_scores_temp = checkpoint(
                self.inner_forward,
                article_sentences,
                article_sentences_lengths,
                num_codes_temp,
                codes_temp,
                code_description_temp,
                code_description_length_temp,
                linearized_codes_temp,
                linearized_codes_lengths_temp,
                *self.parameters())
            scores.append(scores_temp)
            word_level_attentions.append(word_level_attentions_temp)
            traceback_word_level_attentions.append(traceback_word_level_attentions_temp)
            sentence_level_scores.append(sentence_level_scores_temp)
        scores = torch.cat(scores, 1)
        word_level_attentions = torch.cat(word_level_attentions, 1)
        traceback_word_level_attentions = torch.cat(traceback_word_level_attentions, 1)
        sentence_level_scores = torch.cat(sentence_level_scores, 1)
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


    def inner_forward(self, article_sentences, article_sentences_lengths, num_codes, codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths, *args):
        codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths = tensor_to_none(codes), tensor_to_none(code_description), tensor_to_none(code_description_length), tensor_to_none(linearized_codes), tensor_to_none(linearized_codes_lengths)
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        article_sentences_lengths, num_codes, encodings, self_attentions, word_level_attentions =\
            article_sentences_lengths.to(self.device2), num_codes.to(self.device2), encodings.to(self.device2), self_attentions.to(self.device2), word_level_attentions.to(self.device2)
        # b, ns, nt = word_level_attentions.shape
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(
            self_attentions.mean(3).view(b*ns, nl, nt, nt),
            attention_vecs=word_level_attentions.view(b*ns, 1, nt))\
            .view(b, ns, nt)
        if codes is None and code_description is None and linearized_codes is None: raise Exception
        all_code_embeddings = []
        if codes is not None:
            codes = codes.to(self.device2)
            all_code_embeddings.append(self.code_embeddings(codes))
        if code_description is not None:
            all_code_embeddings.append(self.clinical_bert_sentences(
                code_description,
                code_description_length,
            )[0].to(self.device2))
        if linearized_codes is not None:
            all_code_embeddings.append(self.linearized_code_transformer(
                linearized_codes,
                linearized_codes_lengths
            )[0])
        if self.linear2 is not None:
            code_embeddings = torch.cat(all_code_embeddings, 2)
            code_embeddings = self.linear2(code_embeddings)
        else:
            code_embeddings = all_code_embeddings[0]
        sentence_level_scores = self.predict_targets(code_embeddings.transpose(0, 1), encodings.transpose(0, 1)).transpose(0, 2).transpose(1, 2)
        mask = (article_sentences_lengths != 0)[:,:encodings.size(1)].unsqueeze(1).expand(sentence_level_scores.shape)
#        sentence_level_scores_masked = sentence_level_scores*mask
        sentence_level_scores_masked = torch.zeros_like(sentence_level_scores).masked_scatter(mask, sentence_level_scores[mask])
        scores = sentence_level_scores_masked.sum(2)/mask.sum(2)
        requires_grad = scores.requires_grad
        return scores, word_level_attentions.requires_grad_(requires_grad), traceback_word_level_attentions.requires_grad_(requires_grad), sentence_level_scores


def loss_func(total_num_codes, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels):
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

def statistics_func(total_num_codes, scores, codes, num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels):
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
