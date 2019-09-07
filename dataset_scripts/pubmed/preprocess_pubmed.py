from tqdm import tqdm
import random
import os
import pickle as pkl
import pandas as pd
from evidence_inference.preprocess.article_reader import Article
from evidence_inference.preprocess.sentence_split import split
from evidence_inference.preprocess.preprocessor import extract_raw_text, train_document_ids, get_inference_vectorizer

def main(datadir):
    inference_vectorizer = get_inference_vectorizer(train_document_ids(), vocabulary_file=os.path.join("./annotations", "vocab.txt"))
    print("got inference vectorizer")
    rows = []
    random.seed(0)
    xml_list = list_xml_path(datadir)
    random.shuffle(xml_list)
    xml_list = xml_list[:10000]
#    xml_list = xml_list[:100]
    for i,p in tqdm(enumerate(xml_list), total=len(xml_list)):
        try:
            a = Article(p)
            article_text = extract_raw_text(a)
            sentences = split(article_text)
            rows.append({'article_sentences':[sent for sent in sentences]})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    print('read in articles')
    df = df.sample(frac=1)
    df_train = df[:int(len(df)*.8)]
    df_val = df[int(len(df)*.8):int(len(df)*.9)]
    df_test = df[int(len(df)*.9):]
    df_train.to_json('train_processed.data', orient='records', lines=True, compression='gzip')
    df_val.to_json('val_processed.data', orient='records', lines=True, compression='gzip')
    df_test.to_json('test_processed.data', orient='records', lines=True, compression='gzip')

# taken from https://github.com/titipata/pubmed_parser/blob/master/pubmed_parser/pubmed_oa_parser.py
def list_xml_path(path_dir):
    fullpath = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path_dir)) for f in fn]
    path_list = [folder for folder in fullpath if os.path.splitext(folder)[-1] in ('.nxml', '.xml')]
    return path_list

if __name__ == '__main__':
    main('../../data/pubmed/data')
