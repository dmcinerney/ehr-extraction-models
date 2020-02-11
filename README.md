# ehr-extraction-models
This repository contains a number of models for performing conditioned extractive summarization on electronic health records.

## Model types

There are a few models contained in the repository. These models use code supervision to train their extractor:

### Sentence-level Attention

This model takes text and code, produces an attention distribution (used as the extraction distribution), and uses this to predict a final binary prediction of whether or not the code will appear in the patient's future.

### Sentence-level Prediction

This model takes the text and code, but produces predictions for each sentence, softmaxing the predicted scores to produce the extraction distribution over sentences.

### Random Sentence-level model (...coming soon)

This model produces the extraction distribution randomly from a uniform dirichlet?

## Code Embeddings

The first two of the above models uses an embedding to encode the code.  These embeddings can come from 3 different tequniques:

1. Trainable Embedding (adds no bias)
2. Description Embedding (encodes the description using Clinical BERT)
3. Hierarchy Embedding (encodes the path from the top of the code hierarchy to the code)

We train models using these embedding techniques to input bias into the models.

## Instructions

### Setup

Install the requirements:

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

In order to use the code as it currently is, you will also need to follow the intructions in `models/clinical_bert/README.md` to download a pretrained bert on clinical data.  Then update the path variables at the beginning of the `parameter.py` script.

### Datasets

Follow the instructions in https://github.com/dmcinerney/preprocessing-ehr to download and preprocess the MIMIC-III dataset.

### Training

To train the model:

    python train.py [-h] [--data_dir DATA_DIR] [--code_graph_file CODE_GRAPH_FILE]
                    [--save_checkpoint_folder SAVE_CHECKPOINT_FOLDER]
                    [--load_checkpoint_folder LOAD_CHECKPOINT_FOLDER]
                    [--device DEVICE]
                    model_type

### Testing

To test the model you have trained:

    python test.py [-h] [--data_dir DATA_DIR] [--device DEVICE]
                   model_type checkpoint_folder

This will print running statistics to the command line, and the last print will be the final statistics.  It will also save these in the `checkpoint_folder` in `scores.txt`.

### Interface

We also provide an easy interface (that can be used by a server for instance) in order to output the results of one ICD code query on one report.  The result outputs the score (> 0 predicts that the code will occur), heatmaps (a dictionary of extraction distributions over each of the tokens in each sentence of the report), and the tokenized version of the input text.

To use the interface, update the codes_file to point to the corresponding code graph file in the data directory and update the model_dirs to contain key value pairs of the form `model_name : (model_type, checkpoint_folder)`.  Then you can do:

    python
    from interface import FullModelInterface

    # create interface (models_to_load is a list of
    # model_name strings for each model you want to load)
    interface = FullModelInterface(models_to_load) 

    # output all codes and their descriptions
    interface.get_descriptions()

    # output hierarchy (dictionary):
    #   "start" - start node of the hierarchy 
    #   "options" - dictionary from node to list of child nodes
    #   "indices" - dictionary from node to index in parent's options list
    #   "parents" - dictionary from node to parent
    interface.get_hierarchy() # outputs dict with

    # tokenize reports, truncating at num_sentences if it is not 'default'
    # if num_sentences is negative, truncates beginning instead of end
    # returns dictionary:
    #   "tokenized_text" - list of tokenized sentences
    #   "sentence_spans" - list of lists of tuples
    #                      each list of tuples corresponds to a report
    #                      each tuple is (sentence_num, start_idx, end_idx)
    #                      where sentence_num is idx in tokenized_sentences
    #                      and start_idx, end_idx is slice of corresponding
    #                      report in "original_reports"
    #   "original_reports" - list of strings corresponding to raw report
    #                        text
    interface.tokenize(reports, num_sentences='default')

    # outputs the queries that the model has been trained on (other queries can be used)
    interface.get_trained_queries(model_name)

    # outputs bool of whether or not the model allows custom natural language queries
    interface.with_custom(model_name)

    # outputs all of the models loaded in the interface
    interface.get_models()

    # outputs result of performing a query on the dataframe of reports using the model
    # the result is a dictionary:
    #   "heatmaps" - a dictionary of different kinds of attention produced by the model
    #                in the form of a list of lists of token-level attention
    #                (in the case of sentence-level attention, just produce 0's for all
    #                of the tokens that aren't first in the sentence)
    #   "scores" (if the model_type produces a score) - the query's score
    interface.query_reports(model_name, reports, query, is_nl=False)
