import pandas as pd
import stumpy
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
This module contains functions used for the processing of cycles 
(in the context of ICS, an instance from the dataset is usually referred to as a cycle). 
'''

def scale_df_data(df):
    '''
    Transform the given one-feature dataframe with numerical values by scaling the values into range [0, 1]
    :param df:
    :return: a flattern (1D) numpy array with dataframe values scaled
    '''
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_data = scaled_data.flatten()
    return scaled_data


def downsampling_df(df, n = 1000, window_size = 5):
    '''
    Downsampling the content of a time series represented as Dataframe: each :param window_size consecutive values from the time series
    :param df are used to compute their mean value; the new obtained values are collected on a new DataFrame
    :param df: pandas Dataframe
    :param n: first n records to be transformed; use the number of dataframe rows if you want to use all values
    :param window_size:
    :return:
    :rtype: pandas Dataframe
    '''
    if n > len(df):
        n = len(df)

    df_selected = df.head(n)

    df_selected['Group'] = np.arange(len(df_selected)) // window_size
    grouped_means = df_selected.groupby('Group')['_value'].mean()

    new_df = pd.DataFrame({'_value': list(grouped_means)})

    return new_df


def match_cycles_from_ts(data_pattern, data_all, std_dist_val = 2.0):
    '''
    Extract cycles (subsequences) from :param data_all (time series) that are similar enough with respect to :param data_pattern.
    Both :param data_pattern and :param data_all must to be numpy arrays; :param std_dist_val refers to value that control
    the matching threshold - it can be constant or relatively; in this case, it depends on the standard deviation computed using the values from
    :param data_pattern and the processed subsequence from the time series. The Matrix profile method works properly, without to assume
    the normal distribution for the values in the time series
    https://stumpy.readthedocs.io/en/latest/Tutorial_Pattern_Matching.html
    :param data_pattern:
    :param data_all:
    :param std_dist_val:
    :return: matched cycles
    :rtype: a numpy array that contains tuples, each tuple contains 2 numpy.float64 values:
    first value is related to distance between data_pattern and matched cycle and second value is the start index of
    the matched cycle in the time series (data_all)
    '''
    data_all_copy = copy.deepcopy(data_all)
    matches = stumpy.match(data_pattern, data_all_copy, max_distance=lambda D: np.nanmax([np.nanmean(D) - std_dist_val * np.nanstd(D), np.nanmin(D)]))
    return matches

def extract_non_overlayed_cycles_indexes(indexes_list, pattern_len):
    '''
    Extract (cycles) indexes such that the obtained cycles do not overlay. E.g: consider cycles' starting
    indexes in the time series:  [3, 28, 53, 78, 79, 80, 200] and pattern len = 24 (all cycles have the same len),
    this means that a cycle start at index 3, another one at index 28 and so; because there are cycles that start
    at indexes 78, 79 this means that those 2 cycles overlay each other - the distance between them should be greater or equal than
    :param pattern_len, so take only the cycle that start at index 78 and ignore cycle that start at index 79.
    :param indexes_list:
    :param pattern_len: cycles length
    :return:
    '''
    if len(indexes_list) == 0:
        return []

    A = copy.deepcopy(indexes_list)
    A.sort()
    resulted_idxs = []
    current = A[0]
    resulted_idxs.append(current)
    i = 1
    n = len(A)
    while i < n:
        if A[i] - current >= pattern_len:
            current = A[i]
            resulted_idxs.append(current)

        i = i + 1

    return resulted_idxs

def extract_cycles_from_ts(indexes_list, pattern_len, data_all):
    '''
    Using the cycles starting indexes (obtained with stumpy), extract cycles from whole time series
    :param indexes_list:
    :param pattern_len:
    :param data_all:
    :return: a list with the extracted cycles
    :rtype: built-in python list that contains numpy arrays of size pattern_len
    '''
    tseries = []
    for idx in indexes_list:
        selected_ts = data_all[idx:idx+pattern_len]
        tseries.append(selected_ts)
    return tseries

def get_values_by_pos(cycles):
    '''
    For each position (index) in standard cycle (all cycle have same length) gather together all values.
    :param cycles: the given cycles / instances (i.e. n-dimensional points)
    :return: a list with numpy arrays, each array contains the values for a certain position
    #:rtype: built-in list of numpy arrays
    '''
    if len(cycles) == 0:
        return []

    cycle_len = len(cycles[0])
    values_by_pos = []

    for pos_index in range(cycle_len):
        values_in_pos_index = np.array([cycle[pos_index] for cycle in cycles ])
        values_by_pos.append(values_in_pos_index)

    return values_by_pos

# OUT OF USAGE: it was used in the past for datasets with features which follow normal distributions
def get_cycle_anomalous_state_and_pos(cycle, mean_vals, std_vals, std_factor = 2, no_of_points_threshold = 12):
    '''
    This method assume normal distribution.
    For a given cycle, determine if it is anomalous or not. Consider cycle to by anomalous if for at least
    :param no_of_points_threshold values of it, the values exceed standard deviation with a factor of :param std_factor.
    E.g if a cycle have length 24, and over 12 points (values) are anomalous (they exceed 2 standard deviation), then it is anomalous
    :param cycle:
    :param mean_vals: mean values for every position in the cycle
    :param std_vals: std values for every position in the cycle
    :param std_factor:
    :param no_of_points_threshold:
    :return: a tuple with 2 values: anomalous state (True or False) and positions (indexes) for all anomalous points founded in the cycle
    '''
    all_positions = range(len(cycle))
    anomalous_points_positions = [pos for pos in all_positions if (np.abs(cycle[pos] - mean_vals[pos]) > std_factor * std_vals[pos])]

    if len(anomalous_points_positions) >= no_of_points_threshold:
        return (True, anomalous_points_positions)

    return (False, anomalous_points_positions)

# OUT OF USAGE: it was used in the past for datasets with features which follow normal distributions
def label_cycles(cycles_list, mean_by_pos, std_by_pos, std_factor = 1.9, no_of_points_threshold = 6):
    '''
    Assign a label to each cycle as follows: 1 means that the cycle is anomalous, 0 means that the cycle is not anomalous.
    Use get_cycle_anomalous_state_and_pos function to decide if the cycle is anomalous or not.
    :param cycles_list:
    :param mean_by_pos:
    :param std_by_pos:
    :param std_factor:
    :param no_of_points_threshold:
    :return: a list of tuples, each tuple contain: cycle itself, anomalous state (True or False) and anomalous points from the cycle
    '''

    # first get all tuples into a list
    res = [(cycle,) + get_cycle_anomalous_state_and_pos(cycle, mean_by_pos, std_by_pos, std_factor, no_of_points_threshold) for cycle in cycles_list]

    # convert True, False to 1, 0
    res = [(item[0], 1, item[2]) if item[1] is True else (item[0], 0, item[2]) for item in res]
    return res

def get_anomalous_points_pos_freq(lists_of_pos, pattern_len):
    '''
    Compute frequencies for occurrences of positions given in sublists from lists_of_pos
    :param lists_of_pos:
    :param pattern_len:
    :return: np array with length of pattern_len, value on ith position represent the frequency of the value i considering all sublists
    '''
    res = np.zeros((pattern_len))

    for anomalous_points_pos in lists_of_pos:
        for pos in anomalous_points_pos:
            res[pos] = res[pos] + 1

    return res
