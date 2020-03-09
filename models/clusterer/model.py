import torch
from torch import nn

class Clusterer(nn.Module):
    def __init__(self):
        super(Clusterer, self).__init__()

    def forward(self, article_sentences, article_sentences_lengths, attention, num_codes):
        # 1-sentence clusters
        #clustering_lengths = (article_sentences_lengths == 0).int()
        #import pdb; pdb.set_trace()
        #return article_sentences.unsqueeze(-1), clustering_lengths
        # exact-match clusters
        mask = article_sentences_lengths == 0
        num_sentences = (article_sentences_lengths != 0).sum(1)
        sentence_attention = attention.sum(-1)
        sentence_attention[mask.unsqueeze(1).expand(sentence_attention.shape)] = -1
        sorted_indices = torch.argsort(-sentence_attention)
        #similarity = (article_sentences.to(self.device) @ article_sentences.to(self.device).transpose(-1, -2)).unsqueeze(-1)
        clustering = []
        for b in range(sorted_indices.size(0)):
            clustering.append([])
            for c in range(num_codes[b]):
                clustering[-1].append([])
                groups = {}
                for i in range(num_sentences[b]):
                    s = sorted_indices[b, c, i]
                    sentence = tuple(article_sentences[b, s].tolist())
                    if sentence not in groups.keys():
                        groups[sentence] = len(groups)
                        clustering[-1][-1].append([])
                    clustering[-1][-1][groups[sentence]].append(s.item())
        return clustering
