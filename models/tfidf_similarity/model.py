import torch

class Model:
    def __init__(self, num_codes):
        self.num_codes = num_codes

    def correct_devices(self):
        pass

    def forward(self, article_sentences, article_sentences_lengths, num_codes, code_description, code_description_length):
        attention = None
        traceback_attention = None
        """
        B = batch size
        S = max number of sentences per instance
        T_S = max number of tokens per sentence
        C = max number of codes per instance
        T_C = max number of tokens per code description
        INPUTS:
            article_sentences (report text): (B, S, T_S)
            article_sentences_lengths (length of each sentence, where padding sentences are given a length 0): (B, S)
            num_codes (number of codes for each instance, probably don't need this): (B,)
            code_description (description text): (B, C, T_C)
            code_description (length of description text, where padding code descriptions are given a length 0): (B, C)

        TODO CHARLIE: implement function
             Produce attention and traceback attention pytorch tensors both of shape (B, S, C, T_S)

        NOTE: The way I formualted it initially, you will have to return per token attention of size (B, S, C, T_S)
              instead of per sentence attention of size (B, S, C).  This is to be more general in case later on we
              would like per token attention.  For now, just set the attention on the first token in the sentence
              to the sentence attention and the attention of all other tokens to 0.  Also, don't worry about traceback
              attention! Just set it to be the same as the attention!
              Example attention with one instance in the batch:
                   [[[.71, 0, 0, 0],
                     [.04, 0, 0, 0],
                     [.17, 0, 0, 0],
                     [.08, 0, 0, 0]]]
        """
        return_dict = dict(
            num_codes=num_codes,
            attention=attention,
            traceback_attention=traceback_attention,
            article_sentences_lengths=article_sentences_lengths)
        return return_dict
