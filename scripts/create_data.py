import pandas as pd
import numpy as np
import re
import pickle

def read_data(relative_path_to_file):

    """
    Function to return an initial dataframe with cleaned character names, null quotes removed, and lowercase text

    Args:
        relative_path_to_file (str): path to the raw data file

    Returns:
        scripts_df (Pandas DataFrame): dataframe with cleaned character names and valid quotes

    """

    #read  in initial raw data
    scripts_df = pd.read_csv(relative_path_to_file, index_col='Unnamed: 0')

    # remove nulls and non quotes
    scripts_df = scripts_df[~scripts_df.Dialogue.isnull()]
    scripts_df = scripts_df[~scripts_df.Character.str.contains('^\[')]

    # convert dialogue to lowercase
    scripts_df['text'] = scripts_df.Dialogue.str.lower()

    ## create cleaned character names
    scripts_df['Character'] = scripts_df.Character.str.lower()
    scripts_df['Character_clean'] = np.where((scripts_df['Character'] == 'jerry') | (scripts_df['Character'].str.contains("^jerry's")) | (scripts_df['Character'].str.contains("^jerry \(")), 'jerry'
    , np.where((scripts_df['Character'] == 'george') | (scripts_df['Character'].str.contains("^george's")) | (scripts_df['Character'].str.contains("^george \(")), 'george'
    , np.where((scripts_df['Character'] == 'elaine') | (scripts_df['Character'].str.contains("^elaine's")) | (scripts_df['Character'].str.contains("^elaine \(")), 'elaine'
    , np.where((scripts_df['Character'] == 'kramer') | (scripts_df['Character'].str.contains("^kramer's")) | ( scripts_df['Character'].str.contains("^kramer \(")), 'kramer'
    , np.where((scripts_df['Character'] == 'newman') | (scripts_df['Character'].str.contains("^newman \(")),'newman'
    , np.where((scripts_df['Character'] == 'morty') | (scripts_df['Character'].str.contains("^morty \(")), 'morty'
    , np.where((scripts_df['Character'] == 'helen') | (scripts_df['Character'].str.contains("^helen \(")), 'helen'
    , np.where((scripts_df['Character'] == 'frank') | (scripts_df['Character'].str.contains("^frank \(")), 'frank'
    , np.where((scripts_df['Character'] == 'susan') | (scripts_df['Character'].str.contains("^susan \(")), 'susan'
    , np.where((scripts_df['Character'] == 'estelle') | (scripts_df['Character'].str.contains("^estelle \(")), 'estelle'
    , np.where((scripts_df['Character'] == 'peterman') | (scripts_df['Character'] == 'j. peterman') | (scripts_df['Character'] == 'mr. peterman') | (scripts_df['Character'].str.contains("^peterman \(")), 'peterman'
    , np.where((scripts_df['Character'] == 'puddy'), 'puddy'
    , np.where((scripts_df['Character'] == 'jack'), 'jack'
    , np.where((scripts_df['Character'] == 'mickey'), 'mickey'
    , np.where((scripts_df['Character'] == 'bania'), 'bania'
    , np.where((scripts_df['Character'] == 'wilhelm'), 'wilhelm'
    , np.where((scripts_df['Character'] == 'lloyd'), 'lloyd'
    , np.where((scripts_df['Character'] == 'steinbrenner') | (scripts_df['Character'] == 'mr. steinbrenner') | (scripts_df['Character'].str.contains("^steinbrenner \(")), 'steinbrenner'
    , np.where((scripts_df['Character'] == 'lippman') | (scripts_df['Character'] == 'mr. lippman'), 'lippman'
    , np.where((scripts_df['Character'] == 'leo') | (scripts_df['Character'] == 'uncle leo'),'uncle leo'
    , np.where((scripts_df['Character'] == 'jackie') | (scripts_df['Character'].str.contains("^jackie \(")), 'jackie', 'other')))))))))))))))))))))


    return scripts_df

def create_features(df):

    """
    Function to return a DataFrame with Target (dependent) variable and non tf-idf features created

    Args:
        df (Pandas DataFrame): DataFrame with line by line quotes and character names to build features and labels from

    Returns:
        df (Pandas DataFrame): dataframe with labels and new features created based on quote attributes

    """
    # create fields identifying punctuation
    df['exclam_cnt'] = df['text'].str.count('!')
    df['question_cnt'] = df['text'].str.count('\?')

    # create flags for character name mentiones
    char_list = ['lomez','sacamano','devola','steinbrenner','newman','pitt','peterman','kruger']

    for c in char_list:
        df[c] = df['text'].str.contains(c).astype(int)

    # create fields for character 'wordiness'
    df['quote_len'] = df.text.str.len()
    df['num_words'] = [len(i) for i in df['text'].str.split()]
    df['avg_len_word'] = [np.mean(z) for z in [[len(j) for j in i] for i in df['text'].str.split()]]

    # create label (dependent variable)
    df['character_label'] = np.where(
        df['Character_clean'].isin(['jerry', 'george', 'elaine', 'kramer']), df['Character_clean'],
        'other')

    ## only subset for 4 top characters
    df = df[df['character_label'] != 'other']

    return df

def save_raw_text(df,save_path):

    """
    Function to save text column of data frame to a file to be pre_processed in next steps of the modeling pipeline

    Args:
        df (Pandas DataFrame): dataframe with features and labels

    Returns:
        None
    """

    # save text coumn to a list to be further pre processed
    text_to_save = df['text'].tolist()

    pickle.dump(text_to_save, open(save_path, "wb"))


if __name__ == "__main__":

    """ when this script is called, created initial data frame with labels and features and also saved text to a list to be pre-processed later"""

    data = read_data('../data/scripts.csv')

    data_new = create_features(data)
    data_new.to_csv('../artifacts/initial_data.csv',index=False)

    save_raw_text(data_new, '../artifacts/raw_text.pkl')











