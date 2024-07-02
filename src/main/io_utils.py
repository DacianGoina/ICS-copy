import pandas as pd
import pickle
import numpy as np

'''
This module contains functions that facilitate the reading / writing operations from files.
'''

def import_csv_as_df(file_path):
    '''
    Read the csv flie from the given path and return the result as a pandas Dataframe
    :param file_path:
    :return:
    '''

    df = pd.read_csv(file_path)
    return df


def export_df_as_csv(df, file_path):
    '''
    Export the content of the given DataFrame as a csv file and save it to the given path.
    :param df:
    :param file_path:
    :return:
    '''
    df.to_csv(file_path, index=False, encoding='utf-8')

def import_pkl_obj(file_path):
    '''
    Import the binary object from the given file path. Obs: the result of the import should be a numpy array
    :param file_path:
    :return:
    '''
    try:
        with open(file_path, mode = 'rb') as file:
            res_obj = pickle.load(file)
            return res_obj
    except BaseException as e:
        print(e)
        return None

def export_pkl_obj(obj, file_path):
    '''
    Create a binary object from the given object and save it as a pickle binary file (.pkl) to the given path.
    :param binary_object:
    :param file_path:
    :return:
    '''
    if file_path.endswith(".pkl") is False:
        file_path = file_path + ".pkl"

    with open(file_path, mode = 'wb') as file:
        pickle.dump(obj, file)
