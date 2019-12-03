import pandas as pd
import numpy as np
import pickle
import json

from sklearn.svm import LinearSVC

from pre_process import word_tok, normalize_text, stem

from model_pre_training import tfidf



def create_nonText_features(text_input):
    """function to create non tf-idf features on a new string of text for prediction"""
    # puntuation features
    exclam_cnt = text_input.count('!')
    question_cnt = text_input.count('\?')

    # character mention features
    lomez = int('lomez' in text_input)
    sacamano = int('sacamano' in text_input)
    devola = int('devola' in text_input)
    steinbrenner = int('steinbrenner' in text_input)
    newman = int('newman' in text_input)
    pitt = int('pitt' in text_input)
    peterman = int('peterman' in text_input)
    kruger = int('kruger' in text_input)

    # create character wordiness
    quote_len = len(text_input)
    num_words = len(text_input.split())
    avg_len_word = np.mean([len(i) for i in text_input.split()])

    # create data frame of non text features
    df_non_text = pd.DataFrame({'lomez': [lomez], 'sacamano': [sacamano], 'devola': [devola], 'steinbrenner': [steinbrenner]
                                   , 'newman': [newman], 'pitt': [pitt], 'peterman': [peterman], 'kruger': [kruger]
                                   , 'exclam_cnt': [exclam_cnt], 'question_cnt': [question_cnt], 'quote_len': [quote_len], 'num_words': [num_words], 'avg_len_word': [avg_len_word]})


    return  df_non_text


def process_text(text_input):
    """ function to run pre-processing steps on new string for predictions"""
    text_list = [text_input]

    # tokenize, normalize, and stem text
    words = word_tok(text_list)
    normal = normalize_text(words)
    stems = stem(normal)

    stems_predict = [' '.join(i) for i in stems]

    # load best tf-idf model and fit on new string
    with open('../artifacts/tfidf_1000_feat.pkl', 'rb') as f:
        tf_idf = pickle.load(f)

    tf_idf_feat = tf_idf.transform(stems_predict)

    # create sparse dataframe from tf-idf features
    sdf = pd.SparseDataFrame(tf_idf_feat,
                             columns=tf_idf.get_feature_names(),
                             default_fill_value=0)


    return sdf


def model_data(text_df, nontext_df):
    """ creates combined df with tf-idf and non text features. scales data between 0 and 1"""

    full_data = text_df.merge(nontext_df, how='inner', left_index=True, right_index=True)

    for c in full_data.columns:
        full_data[c] = (full_data[c] - full_data[c].min()) / (full_data[c].max() - full_data[c].min())

    full_data = full_data.fillna(0)

    return full_data


def run_predict(text):
    """ function to run predictions on a new string of text"""

    # load best svm model
    with open('../artifacts/best_svm.pkl','rb') as f:
        svm_model = pickle.load(f)

    text_list = [text]

    # create features and predict
    for t in text_list:
        non_text = create_nonText_features(t)
        text_df = process_text(t)

        fit_data = model_data(text_df, non_text)

        prediction = svm_model.predict(fit_data).tolist()

        return prediction

if __name__ == "__main__":

    text_list = 'this is a test'

    prediction = run_predict(text_list)













