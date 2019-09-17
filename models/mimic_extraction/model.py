import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_bert.model import ClinicalBertSentences


class ClinicalBertMimicExtraction(nn.Module):
    def __init__(self, num_codes, outdim=128):
        super(ClinicalBertMimicExtraction, self).__init__()
        self.clinical_bert_sentences = ClinicalBertSentences(embedding_dim=outdim, conditioned_pool=False, truncate_tokens=100, truncate_sentences=200, sentences_per_checkpoint=5)
        self.code_embeddings = nn.Embedding(num_codes, outdim)
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.linear = nn.Linear(outdim, 1)

    def forward(self, article_sentences, article_sentences_lengths, codes):
        encodings, self_attentions, word_level_attentions = self.clinical_bert_sentences(
            article_sentences, article_sentences_lengths)
        b, ns, nt = word_level_attentions.shape
        code_embeddings = self.code_embeddings(codes)
        encoding, sentence_level_attentions = self.attention(code_embeddings.transpose(0, 1), encodings.transpose(0, 1), encodings.transpose(0, 1))
        nq, _, emb_dim = encoding.shape
        word_level_attentions = word_level_attentions.view(b, 1, ns, nt)\
                                                     .expand(b, nq, ns, nt)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        scores = self.linear(encoding)
        return dict(scores=scores.transpose(0, 1).squeeze(2), attention=attention)

def loss_func(scores, labels, attention):
    positive_labels = labels.sum()
    negative_labels = (labels==labels).sum() - positive_labels
    pos_weight = negative_labels/positive_labels
    return F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight)

def statistics_func(scores, labels, attention):
    return {'positives':(scores > 0).sum(),
            'true_positives':((scores > 0) & (labels == 1)).sum(),
            'relevants':labels.sum(),
            'total_predicted':(scores==scores).sum()}
