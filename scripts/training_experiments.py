import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
import datetime


from model_pre_training import tfidf, non_text_features, get_target, tfidf_data, model_data, split_data

# read in data containing pre-processed data with and without stopwords
stop = pd.read_csv('../artifacts/data_pre-processed.csv')
no_stop = pd.read_csv('../artifacts/data_pre-processed_no_stop.csv')

# create list of dataframe and dataframe names to fit models on
df_list = [stop,no_stop]
df_names = ['stop', 'no_stop']


def logistic_cv(X,y):
    """ function to perform gridsearch CV for logistic regression on features (X) and labels (y)"""
    ## DEFINE LOGISTIC MODEL
    log = LogisticRegression(penalty='l2', random_state=132, multi_class='ovr')

    ## DEFINE HYPERPARAMETER GRID
    param_grid = {'C': [.01, .1, 1, 10],
                  'class_weight': [None, 'balanced']

                  }

    # define CV scoring metric
    beta_score = make_scorer(fbeta_score, average='micro', beta=.5)
    now = datetime.datetime.now()
    # perform gridsearch on parameters
    grid = GridSearchCV(log, param_grid=param_grid, cv=3,

                        scoring=beta_score)

    # fit the grisearch CV
    grid.fit(X, y)

    took = datetime.datetime.now()-now

    #obtain best params
    best_param = grid.best_params_
    # obtain CV score from best params
    cv_score = grid.best_score_

    model_name = 'LOGISTIC'

    return took, best_param, cv_score, model_name


def svm_cv(X, y):

    """ function to perform gridsearch CV for SVM on features (X) and labels (y)"""
    ## DEFINE SVM MODEL
    svm = LinearSVC(penalty='l2',random_state=132)

    ## DEFINE HYPERPARAMETER GRID
    param_grid = {'C': [.01, .1, 1, 10],

                  'class_weight': ['balanced', None]

                  }
    # define CV scoring metric
    beta_score = make_scorer(fbeta_score, average='micro', beta=.5)

    now = datetime.datetime.now()
    # perform gridsearch on parameters
    grid = GridSearchCV(svm, param_grid=param_grid, cv=3,

                        scoring=beta_score)
    # fit the gridsearch CV
    grid.fit(X, y)

    took = datetime.datetime.now() - now
    # obtain optimal params
    best_param = grid.best_params_
    # obtain CV score from params
    cv_score = grid.best_score_

    model_name = 'SVM'

    return took, best_param, cv_score, model_name



if __name__ == "__main__":

    ## perform logistic and SVM gridsearch CV on data with/without stop words and different # of tf-idf features and print results to a file
    with open('../artifacts/experiments_results.txt', 'w', encoding="utf-8") as f:
        for d in range(len(df_list)): # for stop words included/excluded

            print(f'\nCV RESULTS for {df_names[d]} words included\n',file=f)


            num_features_list = [500, 1000] # number of tf-idf features to test

            for n in num_features_list:
                print(f'{n} tf-idf features used\n', file=f)
                # extract tf-idf features for stop words included/excluded
                if df_names[d] == 'no_stop':
                    tfidf_feat, tfidf_names = tfidf((1, 2), df_list[d],n,stop=True)
                else:
                    tfidf_feat, tfidf_names = tfidf((1, 2), df_list[d], n, stop=False)

                # create non text features, labels, and tf-idf features data
                df_nonText = non_text_features(df_list[d])
                y_data = get_target(df_list[d])
                text_df = tfidf_data(tfidf_feat,tfidf_names)

                # create features dataframe combining tf-idf and non-text features
                all_data = model_data(text_df, df_nonText)

                # create train and test splits and output train and test sets to csv
                X, y = split_data(all_data,y_data)
                for split in X:
                    X[split].to_csv(f'../artifacts/{split}_features_{df_names[d]}_{n}_features.csv',index=False)
                for split in y:
                    pd.DataFrame(y[split]).to_csv(f'../artifacts/{split}_labels_{df_names[d]}_{n}_features.csv',index=False)


                # Logistic Gridsearch CV
                took_log, best_param_log, cv_score_log, model_name_log = logistic_cv(X['train'],y['train'])

                print(f'took {took_log} for CV {model_name_log} tuning\n', file=f)
                print(f'best parameters found: {best_param_log}\n', file = f)
                print(f'CV f beta score for best params: {cv_score_log}\n\n', file = f)

                # SVM Gridsearch CV
                took_svm, best_param_svm, cv_score_svm, model_name_svm = svm_cv(X['train'],y['train'])

                print(f'took {took_svm} for CV {model_name_svm} tuning\n', file=f)
                print(f'best parameters found: {best_param_svm}\n', file = f)
                print(f'CV f beta score for best params: {cv_score_svm}\n\n', file = f)








