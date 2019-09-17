import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import ClinicalBertSentences


class ClinicalBertEvidenceInference(nn.Module):
    def __init__(self, outdim=128):
        super(ClinicalBertEvidenceInference, self).__init__()
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, conditioned_pool=True, truncate_tokens=100, truncate_sentences=200, sentences_per_checkpoint=5)
        self.reduce_conditioning_dim = nn.Linear(3*outdim, outdim)
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.linear = nn.Linear(outdim, 3)

    def forward(self, article_sentences, article_sentences_lengths, I, C, O, I_length, C_length, O_length):
        I_encoding, _, _ = self.clinical_bert_sentences(
            I.unsqueeze(1), I_length.unsqueeze(1))
        I_encoding = I_encoding.squeeze(1)
        C_encoding, _, _ = self.clinical_bert_sentences(
            C.unsqueeze(1), C_length.unsqueeze(1))
        C_encoding = C_encoding.squeeze(1)
        O_encoding, _, _ = self.clinical_bert_sentences(
            O.unsqueeze(1), O_length.unsqueeze(1))
        O_encoding = O_encoding.squeeze(1)
        conditioning = self.reduce_conditioning_dim(torch.cat([I_encoding, C_encoding, O_encoding], 1))
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths, conditioning=conditioning)
        encoding, sentence_level_attentions = self.attention(conditioning.unsqueeze(0), encodings.transpose(0, 1), encodings.transpose(0, 1))
        encoding = encoding.squeeze(0)
        sentence_level_attentions = sentence_level_attentions.squeeze(1)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(2)
        y_logprobs = F.log_softmax(self.linear(encoding), 1)
        return dict(y_logprobs=y_logprobs, attention=attention)

def loss_func(y_logprobs, y, attention, evidence):
    return -y_logprobs[torch.arange(y.shape[0]).unsqueeze(1), y.unsqueeze(1)+1].sum()

def statistics_func(y_logprobs, y, attention, evidence):
    likelihood = torch.exp(y_logprobs[torch.arange(y.shape[0]).unsqueeze(1), y.unsqueeze(1)+1]).sum()
    predictions = torch.argmax(y_logprobs, 1)
    true_positives_0, positives_0, relevants_0 = class_stats(predictions==0, (y+1)==0)
    true_positives_1, positives_1, relevants_1 = class_stats(predictions==1, (y+1)==1)
    true_positives_2, positives_2, relevants_2 = class_stats(predictions==2, (y+1)==2)
    evidence = evidence.sum(2) > 0
    _, ns, _ = attention.shape
    return {'attention_overlap': attention[evidence[:,:ns]].sum(),
            'likelihood': likelihood,
            'true_positives_0':true_positives_0,
            'positives_0':positives_0,
            'relevants_0':relevants_0,
            'true_positives_1':true_positives_1,
            'positives_1':positives_1,
            'relevants_1':relevants_1,
            'true_positives_2':true_positives_2,
            'positives_2':positives_2,
            'relevants_2':relevants_2}

def class_stats(predictions, labels):
    true_positives = (predictions & labels).sum()
    positives = predictions.sum()
    relevants = labels.sum()
    return true_positives, positives, relevants
