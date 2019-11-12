# ehr-extraction-models
This repository contains a number of models for performing conditioned extractive summarization on electronic health records.

## Model types

There are a few models contained in the repository. These models use code supervision to train their extractor:

### Sentence-level Attention

This model takes text and code, produces an attention distribution (used as the extraction distribution), and uses this to predict a final binary prediction of whether or not the code will appear in the patient's future.

### Sentence-level Prediction

This model takes the text and code, but produces predictions for each sentence, softmaxing the predicted scores to produce the extraction distribution over sentences.

### Description Embedding model

This model could have either of the above ones as a base, but simply alters the code embedding by concatenating it with a description embedding (obtained from running a code's description through clinical bert) and running this through a linear layer to create a vector of the original code embedding dimension.  The description embedding can also be used simply in place of the code embedding.

### Random Sentence-level model (...coming soon)

This model produces the extraction distribution randomly from a uniform dirichlet?

## Instructions

### Setup

Install the requirements:

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

In order to use the code as it currently is, you will also need to follow the intructions in `models/clinical_bert/README.md` to download a pretrained bert (hopefully on clinical data).  Then update the path variables at the beginning of the `train_mimic_extraciton.py` and `test_mimic_extraction.py` scripts.

### Datasets

Follow the instructions in https://github.com/dmcinerney/preprocessing-ehr to download and preprocess the MIMIC-III dataset.

### Training

To train the model:

    python <training_script>

### Testing

To test the model you have trained:

    python <testing_script>

This will print running statistics to the command line, and the last print will be the final statistics.

We also provide an easy interface (that can be used by a server for instance) in order to output the results of one ICD code query on one report.  The result outputs the score (> 0 predicts that the code will occur), heatmaps (a dictionary of extraction distributions over each of the tokens in each sentence of the report), and the tokenized version of the input text.

To use the interface, update the codes_file and model_file variables in interface.py to point to the corresponding code graph file in the data directory and the `model_state.tpkl` file in a checkpoint folder.  Then you can do:

    python
    import interface
    interface.get_queries() # outputs all possible queries
    interface.query_text(<TEXT>, <QUERY>) # outputs result of performing that query on that text (replace text and query with strings)
