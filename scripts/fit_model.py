import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.metrics import fbeta_score, classification_report




## read in train and test data for the optimal experiments : stop words included and 1000 tf-idf features
features_train = pd.read_csv('../artifacts/train_features_stop_1000_features.csv')
labels_train = pd.read_csv('../artifacts/train_labels_stop_1000_features.csv')

features_test = pd.read_csv('../artifacts/test_features_stop_1000_features.csv')
labels_test = pd.read_csv('../artifacts/test_labels_stop_1000_features.csv')

def run_svm_fit(X_train,y_train,X_test,y_test):
    """ fit best SVM on the training data and predict on the test data. output model evaluation metrics to a file"""
    svm = LinearSVC(loss='squared_hinge', C=.01, class_weight=None)
    svm.fit(X_train, y_train)

    # save best SVM for preditions on new data
    with open('../artifacts/best_svm.pkl', "wb") as f:
        pickle.dump(svm, f)

    # print evaluation metrics to a file

    y_pred = svm.predict(X_test)

    #print(pd.Series(y_pred).value_counts())

    fb_score = fbeta_score(y_test, y_pred, beta=.5, average='micro')

    report = classification_report(y_test,y_pred)

    with open('../artifacts/model_evaluation.txt',"w") as f:
        print('Evaluation Metrics for Model\n', file = f)
        print(f'f_beta score on the test set: {fb_score}\n\n',file=f)
        print(report,file=f)

if __name__ == "__main__":

    # run the svm fit model function
    run_svm_fit(features_train,labels_train,features_test,labels_test)








