import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.clinical_bert.model import ClinicalBertSentences
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad


class Model(nn.Module):
    def __init__(self, special_tokens, vocab_size, outdim=64, sentences_per_checkpoint=10, truncate_summary=50, device1='cpu', device2='cpu', freeze_bert=True):
        super(Model, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            set_dropout(self.clinical_bert_sentences, 0)
            set_require_grad(self.clinical_bert_sentences, False)
        self.decoder = Decoder(special_tokens, self.clinical_bert_sentences.clinical_bert_wrapper, outdim, vocab_size, truncate_summary, device=device2)

    def correct_devices(self):
        self.clinical_bert_sentences.correct_devices()
        self.decoder.correct_devices()

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
        encodings_length = encodings_length.to(self.device1)
        if summary is not None:
            summary, summary_length = summary.to(self.device1), summary_length.to(self.device1)
        output_dict = self.decoder(encodings, encodings_length, summary=summary, summary_length=summary_length)
        sentence_level_attentions = output_dict['sentence_level_attentions']
        nq = sentence_level_attentions.size(1)
        word_level_attentions = word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt).to(self.device2)
        traceback_word_level_attentions = traceback_word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt).to(self.device2)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions*sentence_level_attentions.unsqueeze(3)
        return dict(
            **output_dict,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths
        )

def loss_func(instance_losses, sentence_level_attentions, decoded_summary_length, attention, traceback_attention, article_sentences_lengths):
    return instance_losses.sum()

def statistics_func(instance_losses, sentence_level_attentions, decoded_summary_length, attention, traceback_attention, article_sentences_lengths):
    mask = (torch.arange(attention.size(1), device=decoded_summary_length.device) < decoded_summary_length.unsqueeze(1))
    b, nq, ns, nt = attention.shape
    return {'attention_entropy':entropy(attention.view(b, nq, ns*nt))[mask].mean()*b,
            'traceback_attention_entropy':entropy(traceback_attention.view(b, nq, ns*nt))[mask].mean()*b}

class Decoder(nn.Module):
    def __init__(self, special_tokens, clinical_bert_wrapper, outdim, vocab_size, truncate_summary, device='cpu'):
        super(Decoder, self).__init__()
        self.start_token_id = special_tokens['start']
        self.stop_token_id = special_tokens['stop']
        self.mask_token_id = special_tokens['mask']
        self.clinical_bert_wrapper = clinical_bert_wrapper
#        self.embedding = nn.Embedding(vocab_size, outdim)
#        self.lstm = nn.LSTM(outdim, outdim, batch_first=True)
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.linear1 = nn.Linear(self.clinical_bert_wrapper.hidden_size, outdim)
#        self.linear1 = nn.Linear(2*outdim, outdim)
        self.linear2 = nn.Linear(2*outdim, vocab_size)
        self.truncate_summary = truncate_summary
        self.device = device

    def correct_devices(self):
        self.attention = self.attention.to(self.device)
        self.linear1 = self.linear1.to(self.device)
        self.linear2 = self.linear2.to(self.device)

    def forward(self, text_states, text_length, summary=None, summary_length=None, beam_size=1):
        if summary is None:
            return self.decode_generate(text_states, text_length, beam_size=beam_size)
        else:
            return self.decode_train(text_states, text_length, summary, summary_length)

    def decode_generate(self, text_states, text_length, beam_size=1):
        if beam_size > 1:
            raise NotImplementedError
        sentence_level_attentions = []
        summary = torch.zeros_like(text_length).unsqueeze(1) + self.start_token_id
        instance_losses = torch.zeros(summary.size(0), device=self.device)
        valid_indices = torch.arange(summary.size(0), device=self.device)
        for t in range(self.truncate_summary-1):
            summary_masked = torch.cat((summary[valid_indices], torch.zeros_like(summary[valid_indices,:1])+self.mask_token_id), 1)
            attention_mask = torch.ones_like(summary_masked)
            vocab_dist, sentence_level_attention_temp = self.timestep(summary_masked, attention_mask, text_states[valid_indices], text_length[valid_indices])
            predicted_temp = vocab_dist.argmax(1)
            predicted = (torch.zeros_like(summary[:,0])+self.stop_token_id).scatter(0, valid_indices, predicted_temp)
            losses_temp = -torch.log(vocab_dist[torch.arange(vocab_dist.size(0)), predicted_temp])
            instance_losses = instance_losses.scatter_add(0, valid_indices, losses_temp)
            summary = torch.cat((summary, predicted.unsqueeze(1)), 1)
            sentence_level_attention = torch.zeros(
                (text_states.size(0), *sentence_level_attention_temp.shape[1:]),
                device=sentence_level_attention_temp.device)
            sentence_level_attention[valid_indices] = sentence_level_attention_temp
            sentence_level_attentions.append(sentence_level_attention)
            valid_indices = torch.nonzero(predicted != self.stop_token_id)[:,0]
            if len(valid_indices) == 0:
                break
        sentence_level_attentions = torch.cat(sentence_level_attentions, 1)
        decoded_summary_length = summary.size(1)-((summary == self.stop_token_id).sum(1)-1).clamp(0).to(self.device)
        return {
            'instance_losses':instance_losses/decoded_summary_length,
            'sentence_level_attentions':sentence_level_attentions,
            'decoded_summary_length':decoded_summary_length,
            'decoded_summary':summary,
        }

    def decode_train(self, text_states, text_length, summary, summary_length):
        # initialize
        summary = summary[:,:self.truncate_summary]
        sentence_level_attentions = []
        instance_losses = torch.zeros(summary.size(0), device=self.device)
        for t in range(1,summary.size(1)-1):
            valid_indices = torch.nonzero(summary_length > t)[:,0].to(self.device)
            partial_summary = torch.cat((summary[valid_indices,:t], torch.zeros_like(summary[valid_indices,:1])+self.mask_token_id), 1)
            attention_mask = torch.ones_like(partial_summary)
            vocab_dist, sentence_level_attention_temp = checkpoint(self.timestep, partial_summary, attention_mask, text_states[valid_indices], text_length[valid_indices], *self.parameters())
            losses_temp = -torch.log(vocab_dist[torch.arange(vocab_dist.size(0)), summary[valid_indices,t]])
            instance_losses = instance_losses.scatter_add(0, valid_indices, losses_temp)
            sentence_level_attention = torch.zeros(
                (text_states.size(0), *sentence_level_attention_temp.shape[1:]),
                device=sentence_level_attention_temp.device)
            sentence_level_attention[valid_indices] = sentence_level_attention_temp
            sentence_level_attentions.append(sentence_level_attention)
        sentence_level_attentions = torch.cat(sentence_level_attentions, 1)
        decoded_summary_length = (summary_length-1).clamp(0,self.truncate_summary-1).to(self.device)
        return {
            'instance_losses':instance_losses/decoded_summary_length,
            'sentence_level_attentions':sentence_level_attentions,
            'decoded_summary_length':decoded_summary_length
        }

    def timestep(self, partial_summary, attention_mask, text_states, text_length, *args):
#        partial_summary_embedding = self.embedding(partial_summary)
#        decoder_state = self.lstm(partial_summary_embedding[:,:-1])[1]
#        decoder_state = torch.cat(decoder_state, 2)[0]
        decoder_state = self.clinical_bert_wrapper(partial_summary, attention_mask)[0][:,-1]
        decoder_state, text_states, text_length = decoder_state.to(self.device), text_states.to(self.device), text_length.to(self.device)
        decoder_state = self.linear1(decoder_state)
        keys = text_states.transpose(0, 1)
        key_padding_mask = torch.arange(keys.size(0), device=text_length.device) >= text_length.unsqueeze(1)
        context_vector, sentence_level_attention_temp = self.attention(decoder_state.unsqueeze(0), keys, keys, key_padding_mask=key_padding_mask)
        context_vector = context_vector[0]
        encoding = torch.cat((decoder_state, context_vector), 1)
        vocab_dist = F.softmax(self.linear2(encoding), 1)
        return vocab_dist, sentence_level_attention_temp
