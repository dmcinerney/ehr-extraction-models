from transformers import BertTokenizer
from typing import List, Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from abc import ABC


class GenTokenizer(ABC):
    """Abstract class for a tokenizer."""

    @abstractmethod
    def tokenize(self, docs: List[str]) -> List[List[str]]:
        pass

    @abstractmethod
    def append_sent(self, f: Callable[[List[List[str]], List[str]], None],
                    tok_sents: List[List[str]],
                    tok_sent: List[str]) -> None:
        pass


class TfidfTokenizer(GenTokenizer):
    """Tokenizes notes to be used by tfidf."""

    def tokenize(self, docs: List[str]) -> List[List[str]]:
        """Tokenizes the given text into feature vectors via TFIDF."""
        return TfidfVectorizer(input=docs, stop_words='english', vocabulary=None)

    def append_sent(self, f: Callable[[List[List[str]], List[str]], None],
                    tok_sents: List[List[str]],
                    tok_sent: List[str]) -> None:
        """Adds the given tokenized sentence to all tokenized sentences via the given function."""
        f(tok_sents, tok_sent)
