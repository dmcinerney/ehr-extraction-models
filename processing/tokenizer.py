from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pytt.utils import read_pickle
import parameters as p


class AbstractTokenizer:
    """Abstract class for a tokenizer."""
    def __init__(self):
        pass

    def tokenize(self, text):
        pass

    def convert_tokens_to_ids(self, text):
        pass


class TfidfTokenizerWrapper(AbstractTokenizer):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tokenizer = self.vectorizer.build_tokenizer()

    def tokenize(self, text):
        return self.tokenizer(text)

    def convert_tokens_to_ids(self, text_list):
        return self.vectorizer.fit_transform((' '.join(tokenized_text) for tokenized_text in text_list)).toarray()


class BertTokenizerWrapper(AbstractTokenizer):
    def __init__(self, with_cls=True, with_sep=True):
        self.tokenizer = BertTokenizer.from_pretrained(p.pretrained_model)
        self.with_cls = with_cls
        self.with_sep = with_sep

    def tokenize(self, text):
        tokenized_text = []
        if self.with_cls:
            tokenized_text += [self.tokenizer.cls_token]
        tokenized_text += self.tokenizer.tokenize(text)
        if self.with_sep:
            tokenized_text += [self.tokenizer.sep_token]
        return tokenized_text

    def convert_tokens_to_ids(self, text):
        return self.tokenizer.convert_tokens_to_ids(text)
