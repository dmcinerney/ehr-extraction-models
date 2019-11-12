import torch
from pytt.utils import read_pickle
from pytt.testing.raw_individual_processor import RawIndividualProcessor
from models.ehr_extraction.code_supervision.model import Model, loss_func_creator, statistics_func
from models.ehr_extraction.code_supervision.iteration_info import BatchInfo
from models.ehr_extraction.code_supervision_individual_sentence.model import Model as Model_is, loss_func as loss_func_is, statistics_func as statistics_func_is, get_sentence_level_attentions, get_full_attention
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import BatchInfo as BatchInfo_is
from dataset_scripts.ehr.code_dataset.batcher import Batcher

loss_func = loss_func_creator(attention_sparsity=False, traceback_attention_sparsity=False, gamma=1)

class BatchInfoTest(BatchInfo):
    def stats(self):
        self.results, stats = self.test_func(self.batch, **self.batch_outputs)
        return stats

    def filter(self):
        self.batch = None
        self.batch_outputs = None

    def test_func(self, batch, scores, codes, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, labels=None):
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences']}
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
        results = {'scores':scores, 'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences']}
        if labels is not None:
            loss = loss_func_is(scores, codes, num_codes, total_num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels)
            stats = statistics_func_is(scores, codes, num_codes, total_num_codes, word_level_attentions, traceback_word_level_attentions, sentence_level_scores, article_sentences_lengths, labels)
            stats = {'loss': loss, **stats}
        else:
            stats = {}
        return results, stats

model_classes = {
    'code_supervision':(Batcher, Model, BatchInfoTest),
    'code_supervision_individual_sentence':(Batcher, Model_is, BatchInfoTest_is),
}

class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, batch_info_class):
        super(GenericProcessor, self).__init__(model, batcher, batch_info_class=batch_info_class)

    def process_datapoint(self, reports_text, code, label=None):
        raw_datapoint = {'reports':reports_text, 'targets':[code]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, code_graph_file, model_file, model_type):
        batcher_class, model_class, batch_info_class = model_classes[model_type]
        batcher = batcher_class(read_pickle(code_graph_file))
        model = model_class(len(batcher.code_graph.nodes), sentences_per_checkpoint=17, device1='cuda:1', device2='cpu')
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.correct_devices()
        model.eval()
        super(DefaultProcessor, self).__init__(model, batcher, batch_info_class)
