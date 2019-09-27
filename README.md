# ehr-extraction-models
This repository contains a number of models for performing conditioned extractive summarization on electronic health records.

## Instructions

### Setup

Install the requirements:

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

In order to use the code as it currently is, you will also need to follow the intructions in `models/clinical_bert/README.md` to download and use the clinicalBERT model.  Then update the path variables at the beginning of the `train_mimic_extraciton.py` and `test_mimic_extraction.py` scripts.

### Datasets

Follow the instructions in https://github.com/dmcinerney/preprocessing-ehr to download and preprocess the MIMIC-III dataset.

### Training

To train the model:

    python train_mimic_extraction.py

### Testing

To test the model you have trained:

    python test_mimic_extraction.py

This will print running statistics to the command line, and the last print will be the final statistics.  For now, we do not actually do extraction, just output the attention scores over the input sentences, which are not currently recorded in this test script.

We also provide an easy interface (that can be used by a server for instance) in order to output the results of one ICD code query on one report.  The result outputs the score (> 0 predicts that the code will occur), attention (over each of the tokens in each sentence of the report), and the tokenized version of the input text.

To use the interface, update the codes_file and model_file variables in interface.py to point to the codes.pkl file in the data directory and the model_state.tpkl file in a checkpoint folder.  Then you can do:

    python
    import interface
    interface.get_queries() # outputs all possible queries
    interface.query_text(<TEXT>, <QUERY>) # outputs result of performing that query on that text (replace text and query with strings)
