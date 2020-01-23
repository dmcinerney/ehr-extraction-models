from models.ehr_extraction.code_supervision.iteration_info import BatchInfo
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import BatchInfo as BatchInfo_is
from models.ehr_extraction.cosine_similarity.iteration_info import BatchInfo as BatchInfo_cs
from models.ehr_extraction.code_supervision.model import loss_func_creator, statistics_func
from models.ehr_extraction.code_supervision_individual_sentence.model import loss_func as loss_func_is, statistics_func as statistics_func_is
from models.ehr_extraction.cosine_similarity.model import statistics_func as statistics_func_cs

loss_func = loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1)

class BatchInfoTest(BatchInfo):
    def stats(self):
        self.results, stats = self.test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

    def test_func(self, batch, scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels=None):
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        if labels is not None:
            loss = loss_func(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels)
            stats = statistics_func(scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels)
            stats = {'loss': loss, **stats}
        else:
            stats = {}
        return results, stats

class BatchInfoTest_is(BatchInfo_is):
    def stats(self):
        self.results, stats = self.test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

    def test_func(self, batch, scores, codes, num_codes, total_num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels=None):
        # TODO: make result from batch
        sentence_level_attentions = get_sentence_level_attentions(sentence_level_scores, article_sentences_lengths, torch.ones_like(scores).byte())
        attention = get_full_attention(word_level_attentions, sentence_level_attentions)
        traceback_attention = get_full_attention(traceback_word_level_attentions, sentence_level_attentions)
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        if labels is not None:
            loss = loss_func_is(scores, codes, num_codes, total_num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels)
            stats = statistics_func_is(scores, codes, num_codes, total_num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels)
            stats = {'loss': loss, **stats}
        else:
            stats = {}
        return results, stats

class BatchInfoTest_cs(BatchInfo_cs):
    def stats(self):
        self.results, stats = self.test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

    def test_func(self, batch, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, codes=None):
        results = {'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        if codes is not None:
            stats = statistics_func_cs(num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, codes, labels=labels)
        else:
            stats = {}
        return results, stats
