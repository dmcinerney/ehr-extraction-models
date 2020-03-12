import torch
from torch import nn
from models.clusterer.model import Clusterer

class Model(nn.Module):
    def __init__(self, device='cpu', cluster=False):
        super(Model, self).__init__()
        self.device = device
        self.cluster = cluster
        self.clusterer = Clusterer() if cluster else None

    def correct_devices(self):
        pass

    def forward(self, article_sentences, num_sentences, num_codes, code_description):
        attention = (code_description.to(self.device) @ article_sentences.to(self.device).transpose(-1, -2)).unsqueeze(-1)
        traceback_attention = attention
        article_sentences_lengths = (torch.arange(article_sentences.size(1)) < num_sentences.unsqueeze(-1)).long()
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
