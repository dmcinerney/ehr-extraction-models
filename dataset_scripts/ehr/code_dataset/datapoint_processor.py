import torch
from pytt.utils import read_pickle
from pytt.testing.raw_individual_processor import RawIndividualProcessor
from models.ehr_extraction.code_supervision.model import Model, loss_func_creator, statistics_func
from models.ehr_extraction.code_supervision.iteration_info import BatchInfo
from models.ehr_extraction.code_supervision_individual_sentence.model import Model as Model_is, loss_func as loss_func_is, statistics_func as statistics_func_is, get_sentence_level_attentions, get_full_attention
from models.ehr_extraction.code_supervision_individual_sentence.iteration_info import BatchInfo as BatchInfo_is
from models.ehr_extraction.cosine_similarity.model import Model as Model_cs, statistics_func as statistics_func_cs
from models.ehr_extraction.cosine_similarity.iteration_info import BatchInfo as BatchInfo_cs
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

    def test_func(self, batch, num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, codes, labels=None):
        results = {'attention':attention, 'traceback_attention':traceback_attention, 'article_sentences_lengths':article_sentences_lengths, 'tokenized_text':batch.instances[0]['tokenized_sentences'], 'sentence_spans':batch.instances[0]['sentence_spans'], 'original_reports':batch.instances[0]['original_reports']}
        stats = statistics_func_cs(num_codes, total_num_codes, attention, traceback_attention, article_sentences_lengths, codes, labels=labels)
        return results, stats

model_classes = {
    'code_supervision_aligned_both':(
        '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/aligned/model_state.tpkl',
        lambda num_codes: Batcher(num_codes, instance_type='with_description'),
        lambda *args, **kwargs:Model(*args, **kwargs, device1='cuda:0', device2='cpu', freeze_bert=True, reduce_code_embeddings=True),
        BatchInfoTest),
    'code_supervision_description_only':(
        '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision/description_only/model_state.tpkl',
        lambda num_codes: Batcher(num_codes, instance_type='only_description'),
        lambda *args, **kwargs:Model(*args, **kwargs, device1='cuda:0', device2='cpu', freeze_bert=True, reduce_code_embeddings=False),
        BatchInfoTest),
    'code_supervision_individual_sentence':(
        '/home/jered/Documents/projects/ehr-extraction-models/checkpoints/ehr_extraction_code_supervision_individual_sentence/checkpoint/model_state.tpkl',
        Batcher,
        lambda *args, **kwargs:Model_is(*args, **kwargs, device1='cuda:1', device2='cpu'),
        BatchInfoTest_is),
    'cosine_similarity':(
        None,
        lambda num_codes: Batcher(num_codes, instance_type='only_description'),
        lambda *args, **kwargs:Model_cs(*args, **kwargs, device='cuda:1'),
        BatchInfoTest_cs)
}

class GenericProcessor(RawIndividualProcessor):
    def __init__(self, model, batcher, batch_info_class):
        super(GenericProcessor, self).__init__(model, batcher, batch_info_class=batch_info_class)

    def process_datapoint(self, reports, query, label=None, is_nl=False):
        raw_datapoint = {'reports':reports, 'queries' if is_nl else 'targets':[query]}
        if label is not None:
            raw_datapoint['labels'] = [label]
        return super(GenericProcessor, self).process_datapoint(raw_datapoint)

class DefaultProcessor(GenericProcessor):
    def __init__(self, code_graph_file, model_type):
        model_file, batcher_class, model_class, batch_info_class = model_classes[model_type]
        batcher = batcher_class(read_pickle(code_graph_file))
        model = model_class(len(batcher.code_graph.nodes), sentences_per_checkpoint=17)
        if model_file is not None:
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.correct_devices()
        model.eval()
        super(DefaultProcessor, self).__init__(model, batcher, batch_info_class)

    def takes_nl_queries(self):
        return self.batcher.instance_type == "only_description"
