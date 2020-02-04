import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device

    def correct_devices(self):
        pass

    def forward(self, article_sentences, num_sentences, num_codes, code_description):
        attention = (code_description.to(self.device) @ article_sentences.to(self.device).transpose(-1, -2)).unsqueeze(-1)
        traceback_attention = attention
        article_sentences_lengths = (torch.arange(article_sentences.size(1)) < num_sentences.unsqueeze(-1)).long()
        return_dict = dict(
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths)
        return return_dict
