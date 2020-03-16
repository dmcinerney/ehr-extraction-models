import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import EncoderSentences, ClinicalBertWrapper
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts
from models.clusterer.model import Clusterer


class Model(nn.Module):
    def __init__(self, outdim=64, sentences_per_checkpoint=10, device='cpu', cluster=False):
        super(Model, self).__init__()
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, pool_type="mean", truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device)
        self.device = device
        self.cluster = cluster
        self.clusterer = Clusterer() if cluster else None

    def correct_devices(self):
        self.clinical_bert_sentences.correct_devices()

    def forward(self, article_sentences, article_sentences_lengths, num_codes, code_description, code_description_length):
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        # b, ns, nt = word_level_attentions.shape
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(
            self_attentions.mean(3).view(b*ns, nl, nt, nt),
            attention_vecs=word_level_attentions.view(b*ns, 1, nt))\
            .view(b, ns, nt)
        code_embeddings = self.clinical_bert_sentences(code_description, code_description_length)[0]
        key_padding_mask = (article_sentences_lengths == 0)[:,:encodings.size(1)]
        sentence_level_attentions = code_embeddings @ encodings.transpose(-1,-2)
        sentence_level_attentions = sentence_level_attentions/sentence_level_attentions.sum(2, keepdim=True)
        nq = code_description.shape[1]
        word_level_attentions = word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        traceback_word_level_attentions = traceback_word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions*sentence_level_attentions.unsqueeze(3)
        if self.cluster:
            clustering = self.clusterer(article_sentences, article_sentences_lengths, attention, num_codes)
        else:
            clustering = None
        return_dict = dict(
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths,
            clustering=clustering)
        return return_dict


def statistics_func(total_num_codes, num_codes, attention, traceback_attention, article_sentences_lengths, clustering, codes, labels=None):
    #import pdb; pdb.set_trace()
    b, nq, ns, nt = attention.shape
    code_mask = (torch.arange(codes.size(1), device=codes.device) < num_codes.unsqueeze(1))
    return {'attention_entropy':entropy(attention.view(b, nq, ns*nt))[code_mask].mean()*b,
            'traceback_attention_entropy':entropy(traceback_attention.view(b, nq, ns*nt))[code_mask].mean()*b}
