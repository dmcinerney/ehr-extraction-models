import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import EncoderSentences, ClinicalBertWrapper
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts


class Model(nn.Module):
    def __init__(self, num_codes, outdim=64, sentences_per_checkpoint=10, device='cpu'):
        super(Model, self).__init__()
        self.num_codes = num_codes
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, pool_type="mean", truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device)
        self.device = device

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
        sentence_level_attentions = ((code_embeddings.unsqueeze(-2) - encodings.unsqueeze(-3))**2).sum(-1)
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
        return_dict = dict(
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths)
        return return_dict
