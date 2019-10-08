import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import ClinicalBertSentences
from utils import traceback_attention as ta


class Model(nn.Module):
    def __init__(self, num_codes, outdim=64, sentences_per_checkpoint=10):
        super(Model, self).__init__()
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, conditioned_pool=False, truncate_tokens=100, truncate_sentences=100, sentences_per_checkpoint=sentences_per_checkpoint)
        self.decoder = Decoder(self.clinical_bert_sentences.clinical_bert_wrapper)
        self.linear = nn.Linear(outdim, 1)

    def forward(self, article_sentences, article_sentences_lengths, summary=None, summary_length=None):
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        # b, ns, nt = word_level_attentions.shape
        b, ns, nl, nh, nt, _ = self_attentions.shape
        traceback_word_level_attentions = ta(
            self_attentions.mean(3).view(b*ns, nl, nt, nt),
            attention_vecs=word_level_attentions.view(b*ns, 1, nt))\
            .view(b, ns, nt)
        encodings_length = (article_sentences_lengths > 0).sum(1)
        output_dict = self.decoder(encodings, encodings_length, summary=summary, summary_length=summary_length)
        sentence_level_attentions = output_dict['sentence_level_attentions']
        nq = sentence_level_attentions.size(1)
        word_level_attentions = word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        traceback_word_level_attentions = traceback_word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions*sentence_level_attentions.unsqueeze(3)
        return dict(
            **output_dict,
            attention=attention,
            traceback_attention=traceback_attention,
        )

def loss_func(instance_losses, sentence_level_attentions, attention, traceback_attention):
    pass

def statistics_func(instance_losses, sentence_level_attentions, attention, traceback_attention):
    pass

class Decoder(nn.Module):
    def __init__(self, clinical_bert_wrapper):
        super(Decoder, self).__init__()
        self.clinical_bert_wrapper = clinical_bert_wrapper

    def forward(self, text_states, text_length, summary=None, summary_length=None, beam_size=1):
        if summary is None:
            return self.decode_generate(text_states, text_length, beam_size=beam_size)
        else:
            return self.decode_train(text_states, text_length, summary, summary_length)

    def decode_generate(self, text_states, text_length, state, beam_size=1):
        raise NotImplementedError

    def decode_train(text_states, text_length, summary, summary_length):
        # initialize
        for t in range(summary.size(1)-1):
            import pdb; pdb.set_trace()
        return {
            'instance_losses':None,
            'sentence_level_attentions':None
        }
