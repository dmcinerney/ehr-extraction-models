import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.clinical_bert.model import ClinicalBertWrapper, EncoderSentences
from models.clusterer.model import Clusterer
from utils import traceback_attention as ta, entropy, set_dropout, set_require_grad, get_code_counts, tensor_to_none


class Model(nn.Module):
    def __init__(self, outdim=64, sentences_per_checkpoint=10, device1='cpu', device2='cpu', freeze_bert=True, code_embedding_type_params=set([]), concatenate_code_embedding=False, dropout=.15, cluster=False):
        super(Model, self).__init__()
        self.clinical_bert_sentences = EncoderSentences(ClinicalBertWrapper, embedding_dim=outdim, truncate_tokens=50, truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device1)
        if freeze_bert:
            self.freeze_bert()
        else:
            self.unfreeze_bert(dropout=dropout)
        self.code_embedding_type_params = code_embedding_type_params
        num_code_embedding_types = len(code_embedding_type_params)
        self.code_embeddings = nn.Embedding(code_embedding_type_params['codes'][0], outdim)\
                               if 'codes' in code_embedding_type_params.keys() else None
        self.linearized_code_transformer = EncoderSentences(lambda : LinearizedCodesTransformer(num_linearization_embeddings), embedding_dim=outdim, truncate_tokens=50,
                                                            truncate_sentences=1000, sentences_per_checkpoint=sentences_per_checkpoint, device=device2)\
                                           if 'linearized_codes' in code_embedding_type_params.keys() else None
        self.attention = nn.MultiheadAttention(outdim, 1)
        self.concatenate_code_embedding = concatenate_code_embedding
        self.linear = nn.Linear(outdim, 1)
        self.linear2 = nn.Linear(outdim*num_code_embedding_types, outdim) if num_code_embedding_types > 1 else None
        self.linear3 = nn.Linear(2*outdim, outdim) if concatenate_code_embedding else None
        self.device1 = device1
        self.device2 = device2
        self.cluster = cluster
        self.clusterer = Clusterer() if cluster else None

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
        self.attention.to(self.device2)
        self.linear.to(self.device2)
        if self.linear2 is not None:
            self.linear2.to(self.device2)
        self.codes_per_checkpoint = 1000
        if self.cluster:
            self.clusterer.to(self.device2)
        if self.concatenate_code_embedding:
            self.linear3.to(self.device2)

    def forward(self, article_sentences, article_sentences_lengths, num_codes, codes=None, code_description=None, code_description_length=None, linearized_codes=None, linearized_codes_lengths=None, linearized_descriptions=None, linearized_descriptions_lengths=None):
        nq = num_codes.max()
        nq_temp =  self.codes_per_checkpoint
        scores, attention, traceback_attention, context_vec = [], [], [], []
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
            if linearized_descriptions is not None:
                linearized_descriptions_temp = linearized_descriptions[:, offset:offset+nq_temp]
                linearized_descriptions_lengths_temp = linearized_descriptions_lengths[:, offset:offset+nq_temp]
            else:
                linearized_descriptions_temp = torch.zeros(0)
                linearized_descriptions_lengths_temp = torch.zeros(0)
            num_codes_temp = torch.clamp(num_codes-offset, 0, nq_temp)
            scores_temp, attention_temp, traceback_attention_temp, context_vec_temp = checkpoint(
                self.inner_forward,
                article_sentences,
                article_sentences_lengths,
                num_codes_temp,
                codes_temp,
                code_description_temp,
                code_description_length_temp,
                linearized_codes_temp,
                linearized_codes_lengths_temp,
                linearized_descriptions_temp,
                linearized_descriptions_lengths_temp,
                *self.parameters())
            scores.append(scores_temp)
            attention.append(attention_temp)
            traceback_attention.append(traceback_attention_temp)
            context_vec.append(context_vec_temp)
        scores = torch.cat(scores, 1)
        attention = torch.cat(attention, 1)
        traceback_attention = torch.cat(traceback_attention, 1)
        context_vec = torch.cat(context_vec, 1)
        if self.cluster:
            clustering = self.clusterer(article_sentences, article_sentences_lengths, attention, num_codes)
        else:
            clustering = None
        return_dict = dict(
            scores=scores,
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths,
            clustering=clustering,
            context_vec=context_vec)
        if codes is not None:
            return_dict['codes'] = codes
        return return_dict

    def inner_forward(self, article_sentences, article_sentences_lengths, num_codes, codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths, linearized_descriptions, linearized_descriptions_lengths, *args):
        codes, code_description, code_description_length, linearized_codes, linearized_codes_lengths, linearized_descriptions, linearized_descriptions_lengths = tensor_to_none(codes), tensor_to_none(code_description), tensor_to_none(code_description_length), tensor_to_none(linearized_codes), tensor_to_none(linearized_codes_lengths), tensor_to_none(linearized_descriptions), tensor_to_none(linearized_descriptions_lengths)
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
        if codes is None and code_description is None and linearized_codes is None and linearized_descriptions is None: raise Exception
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
        if linearized_descriptions is not None:
            all_code_embeddings.append(self.clinical_bert_sentences(
                linearized_descriptions,
                linearized_descriptions_lengths,
            )[0].to(self.device2))
        if self.linear2 is not None:
            code_embeddings = torch.cat(all_code_embeddings, 2)
            code_embeddings = self.linear2(code_embeddings)
        else:
            code_embeddings = all_code_embeddings[0]
        key_padding_mask = (article_sentences_lengths == 0)[:,:encodings.size(1)]
        contextvec, sentence_level_attentions = self.attention(code_embeddings.transpose(0, 1), encodings.transpose(0, 1), encodings.transpose(0, 1), key_padding_mask=key_padding_mask)
        nq, _, emb_dim = contextvec.shape
        word_level_attentions = word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        traceback_word_level_attentions = traceback_word_level_attentions\
            .view(b, 1, ns, nt)\
            .expand(b, nq, ns, nt)
        attention = word_level_attentions*sentence_level_attentions.unsqueeze(3)
        traceback_attention = traceback_word_level_attentions*sentence_level_attentions.unsqueeze(3)
        if self.concatenate_code_embedding:
            encoding = torch.cat([contextvec, code_embeddings.transpose(0, 1)], 2)
            encoding = torch.relu(self.linear3(encoding))
        else:
            encoding = contextvec
        scores = self.linear(encoding)
        return scores.transpose(0, 1).squeeze(2), attention, traceback_attention, contextvec.transpose(0, 1)

def abstract_loss_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels, attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    b, nq, ns, nt = attention.shape
    positive_labels = labels.sum()
    negative_labels = num_codes.sum() - positive_labels
    pos_weight = negative_labels/positive_labels
    losses = F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pos_weight, reduction='none')
    code_mask = (torch.arange(nq, device=labels.device) < num_codes.unsqueeze(1))
    loss = losses[code_mask].mean()*b
    if attention_sparsity:
        loss += gamma*entropy(attention.view(b, nq, ns*nt))[code_mask].mean()*b
    if traceback_attention_sparsity:
        loss += gamma*entropy(traceback_attention.view(b, nq, ns*nt))[code_mask].mean()*b
    return loss

def loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1):
    def loss_func_wrapper(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels):
        return abstract_loss_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels,
                                  attention_sparsity=attention_sparsity, traceback_attention_sparsity=traceback_attention_sparsity, gamma=gamma)
    return loss_func_wrapper

def statistics_func(total_num_codes, scores, codes, num_codes, attention, traceback_attention, context_vec, article_sentences_lengths, clustering, labels):
    b, nq, ns, nt = attention.shape
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


class LinearizedCodesTransformer(nn.Module):
    def __init__(self, num_embeddings, d_model=100, num_layers=6, nhead=4):
        super(LinearizedCodesTransformer, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, d_model)
        self.positional_encodings = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.hidden_size = d_model
        self.num_layers = num_layers
        self.num_heads = nhead

    def forward(self, token_ids, attention_mask):
        b, nl, nh, nt = token_ids.size(0), self.num_layers, self.num_heads, token_ids.size(1)
        outputs = self.transformer_encoder(self.positional_encodings(self.embeddings(token_ids).transpose(0, 1)), src_key_padding_mask=~attention_mask).transpose(0, 1)
        return outputs, outputs[0,0,0]*torch.eye(nt).expand(b, nl, nh, nt, nt)


# Taken from https://github.com/pytorch/examples/tree/master/word_language_model
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
