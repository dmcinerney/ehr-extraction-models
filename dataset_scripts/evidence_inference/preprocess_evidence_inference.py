import os
import numpy as np
import pandas as pd
import pickle as pkl
from evidence_inference.preprocess.preprocessor import get_Xy, train_document_ids, test_document_ids, validation_document_ids, get_train_Xy
from gensim.models import KeyedVectors

def preprocess_step1():
    print("preprocess step 1")
    tr_ids, val_ids, te_ids = train_document_ids(), validation_document_ids(), test_document_ids()
    vocab_f = os.path.join("./annotations", "vocab.txt")
    print("loading train data")
    train_Xy, inference_vectorizer = get_train_Xy(tr_ids, sections_of_interest=None, vocabulary_file=vocab_f, include_sentence_span_splits = True)
    vocab = inference_vectorizer.idx_to_str
    print("loading vectors")
    WVs = KeyedVectors.load_word2vec_format('embeddings/PubMed-w2v.bin', binary=True)
    mean = sum(WVs.get_vector(word) for word in WVs.vocab)/len(WVs.vocab)
    embeddings = np.array([WVs.get_vector(word) if word not in [None, '<pad>', '<unk>'] else (mean if word != '<pad>' else np.zeros(WVs.vector_size)) for word in vocab])
#    import pdb; pdb.set_trace()
    print("saving vocab")
    with open('vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)
    print("saving vectors")
    with open('vectors.pkl', 'wb') as f:
        pkl.dump(embeddings, f)
#    pdb.set_trace()
    print("saving train data")
    pd.DataFrame(train_Xy).to_json('train_processed.data', orient='records', lines=True, compression='gzip')
    print("loading val data")
    val_Xy  = get_Xy(val_ids, inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True)
    print("loading test data")
    test_Xy = get_Xy(te_ids, inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True)
    print("saving val data")
    pd.DataFrame(val_Xy).to_json('val_processed.data', orient='records', lines=True, compression='gzip')
    print("saving test data")
    pd.DataFrame(test_Xy).to_json('test_processed.data', orient='records', lines=True, compression='gzip')
    print("done")

def preprocess_step2(file, outfile, section_length):
    print("preprocess step 2")
    print("reading data")
    df = pd.read_json(file, lines=True, compression='gzip')
    rows = []
    print("processing data")
    for i,row in df.iterrows():
        length = len(row['article'])
        import pdb; pdb.set_trace()
        rows.append({
            'p_id':row['p_id'],
            'a_id':row['a_id'],
            'I':row['I'],
            'C':row['C'],
            'O':row['O'],
            'article':[sentence for sentence,label in row['sentence_span']],
            'sentence_labels':[label for sentence,label in row['sentence_span']],
            'y':row['y'][0][0],
            'evidence':row['token_ev_labels']
        })
        #for j,(start,end) in enumerate(row['evidence_spans']):
        #    new_row = {'I':row['I'], 'C':row['C'], 'O':row['O'], 'article':row['article'], 'evidence_span':[start, end], 'p_id':row['p_id'], 'a_id':row['a_id'], 'y':row['y'][j][0], 'label_number':j}
        #    earliest = max(0,end-section_length)
        #    latest = min(start,length-section_length)
        #    if earliest >= latest:
        #        continue
        #    section_start = np.random.randint(earliest,latest)
        #    new_row['article'] = row['article'][section_start:section_start+section_length]
        #    new_row['evidence_span'] = [start-section_start, end-section_start]
        #    rows.append(new_row)
        #    break
    print("saving data")
    pd.DataFrame(rows).to_json(outfile, orient='records', lines=True, compression='gzip')

if __name__ == '__main__':
    preprocess_step1()
    print("step 2 processing train")
    preprocess_step2('train_processed.data', 'train_processed2.data', 400)
    print("step 2 processing val")
    preprocess_step2('val_processed.data', 'val_processed2.data', 400)
    print("step 2 processing test")
    preprocess_step2('test_processed.data', 'test_processed2.data', 400)
    print("done")
