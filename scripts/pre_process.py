import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

## open list containing raw text

with open('../artifacts/raw_text.pkl','rb') as f:
    text_list = pickle.load(f)

## open initial dataset

scripts = pd.read_csv('../artifacts/initial_data.csv')

# word tokenize
def word_tok(corpus,remove_stop=False):

    """ inputs a corpus (list of documents) and returns a nested list of word tokenized phrases for each document

    Args:
        corpus (list): list of documents (character quotes)
        remove_stop (bool): true or false for whether to remove stop words
    Returns:
        corpus_token_word (list): nested list of word tokens for each document (quote)

    """
    corpus_token_word = []
    for c in corpus:
        list_of_words = word_tokenize(c) #word tokenize
        # define stop words
        stop_words = nltk.corpus.stopwords.words('english')
        newStopWords = ['hey','yeah','hello','know','like','right','huh']
        stop_words.extend(newStopWords)
        # remove stop words if remove_stop=True
        if remove_stop:
            list_of_words_clean = [w for w in list_of_words if w not in stop_words]
        # remove extra short tokens
        list_of_words_clean = [w for w in list_of_words if len(w) > 2]
        corpus_token_word.append(list_of_words_clean)

    return corpus_token_word

# normalize text
def normalize_text(token_corpus, remove_stop=False):

    """ inputs a word tokenized corpus and returns a nested list of normalized tokens

    Args:
        token_corpus (list): nested list of work tokenized documents
        remove_stop (bool): true false for whether to remove stop words
    Returns:
        normalized_corpus (list): nested list of normalized work tokens per document

    """
    normalized_corpus = []
    stop_words = nltk.corpus.stopwords.words('english')
    newStopWords = ['hey','yeah','hello','know','like','right','huh']
    stop_words.extend(newStopWords)

    for d in token_corpus:
        norm_doc = []
        for s in d:
            s = s.lower()  # convert to lowercase
            s = re.sub(r'\d+', '', s)  # remove numbers
            s = s.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'))  # remove punctuation
            # remove stop words if true
            if remove_stop:
                if s in stop_words:
                    s = ''
            s = re.sub(r"'", "", s)
            s = s.strip()  # remove extra white spaces
            s = re.sub(r'(?:\n|\s+|\t)', ' ', s)  # remove newline characters, exra spaces, and tabs
            norm_doc.append(s)
        norm_doc = [i for i in norm_doc if i]  # remove empty string tokens
        norm_doc = [i for i in norm_doc if len(i) > 2] # remove short tokens

        normalized_corpus.append(norm_doc)

    return normalized_corpus

def stem(normalized_text):

    """ inputs a normalized corpus and returns a nested list of normalized and stemmed tokens

    Args:
        normalized_text (list) : nested list of normalized tokens

    Returns:
        stems (list): nested list of stemmed and normalized tokens
    """
    ps = PorterStemmer()
    stems = []

    for c in normalized_text:
        # for each normalized token, create a stemmed token
        stem_doc = []
        for w in c:
            st = ps.stem(w)
            stem_doc.append(st)
        stem_doc = [i for i in stem_doc if i]  # remove empty string tokens
        stems.append(stem_doc)

    return stems


def create_df(df,features,stem_list):

    """
    function to create a combined dataframe of non text features and a stemmed/normalized version of the quote

    Args:
        df (pandas dataframe): initial data frame with features
        features (list): features to use
        stem_list (list) : list of stems
    Returns:
        df (pandas dataframe) : pandas data frame with stememd dialogue

    """

    df = df[features] # subset data frame
    df['text_proc'] = [' '.join(i) for i in stem_list] # create column to join stemmed tokens for each row
    df = df[df.text_proc.str.strip() != '']
    df = df[df['num_words'] > 5].reset_index(drop=True) # drop quotes that have 5 or less words

    return df


if __name__ == "__main__":

    feature_list = ['character_label', 'lomez', 'sacamano', 'devola', 'steinbrenner', 'newman', 'pitt', 'peterman', 'kruger',
            'exclam_cnt', 'question_cnt', 'quote_len', 'num_words', 'avg_len_word']

    ## create data with stop words INCLUDED
    words = word_tok(text_list)
    normal = normalize_text(words)
    stems = stem(normal)

    data = create_df(scripts,feature_list,stems)
    data.to_csv('../artifacts/data_pre-processed.csv',index=False)

    ## create data with stop words EXCLUDED

    words_no_stop = word_tok(text_list,remove_stop=True)
    normal_no_stop = normalize_text(words_no_stop,remove_stop=True)
    stems_no_stop = stem(normal_no_stop)

    data_no_stop = create_df(scripts, feature_list, stems_no_stop)
    data_no_stop.to_csv('../artifacts/data_pre-processed_no_stop.csv', index=False)


