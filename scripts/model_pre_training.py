import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2


def tfidf(ngram, df, num_features,stop=False):
    """function to create tf-idf features and feature names

    Args:
        ngram (tuple): tuple for ngram range to pass into TFIDF vectorizer
        df (pandas dataframe) : dataframe with text columns to be fit by tf-idf model
        num_features (int): number of tf-idf features to create
        stop (bool): if true, save a model removing stop words
    Returns:
        features: tf-idf features sparse array
        feature_names: names of the tf-idf features

    """

    # cx`reate tfidf vectorizer
    tf_idf = TfidfVectorizer(sublinear_tf=False, norm='l2', min_df=1,max_features = num_features, ngram_range=ngram, stop_words=None)
    # fit and transform the features to the review data
    features = tf_idf.fit_transform(df.text_proc.tolist())
    feature_names = tf_idf.get_feature_names()

    # save the tfidf features for later use when predicting svm
    model_save = tf_idf.fit(df.text_proc.tolist())

    if stop:
        model_name = f'../artifacts/tfidf_no_stop_{num_features}_feat.pkl'
    else:
        model_name = f'../artifacts/tfidf_{num_features}_feat.pkl'

    pickle.dump(model_save, open(model_name, "wb"))

    return features, feature_names


def non_text_features(df):

  """ returns data frame of non text-based features """

  df = df.drop(['text_proc','character_label'],axis=1)

  df = df.reset_index(drop=True)

  return df


def get_target(df):
    """ returns the dependent variable"""
    target = df.character_label

    return target





def tfidf_data(features, names):
    """ creates sparse dataframe with top N tf-idf features"""

    sdf = pd.SparseDataFrame(features,
                             columns=names,
                             default_fill_value=0)



    return sdf


def model_data(text_df, nontext_df):
    """ creates combined df with tf-idf and non text features. scales data between 0 and 1"""

    full_data = text_df.merge(nontext_df, how='inner', left_index=True, right_index=True)

    for c in full_data.columns:
        full_data[c] = (full_data[c] - full_data[c].min()) / (full_data[c].max() - full_data[c].min())

    full_data = full_data.fillna(0)

    return full_data

def split_data(X,y):

    """ function to create test train splits of the data and returns two dictionaries to store X (features) and y (labels) for both
    train and test
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=132)

    X_data = dict(train=X_train)
    y_labels = dict(train=y_train)

    X_data['test'] = X_test
    y_labels['test'] = y_test

    return X_data, y_labels





