import numpy as np

from src.main.cycle_utils import get_values_by_pos, label_cycles, get_anomalous_points_pos_freq
from src.main.statistics_utils import compute_mean, compute_std, compute_relative_frequencies
from src.model.SlidingWindowItem import SlidingWindowItem
from src.model.ICSEstimator import ICSEstimator
from sklearn.metrics import confusion_matrix

'''
This module contains utility functions that facilitate the creation of various instances (objects) 
for different classes such as window items and ICS estimators.
'''

def get_adjacent_windows_indexes(series_len, window_length = 4):
    '''
    Determine all window item indexes for a given sliding window length and a give series length.
    E.g if series_len = 24 and window_length = 4, then the indexes pairs are: (0,3), (1,4), ..., (20, 23).
    Observation: there exist m = pattern_len - window_size + 1 such possible pairs (window items)
    :param series_len: the length of the series (sequence) for which the window is "moved" to generate subsequences
    :param window_length: window length
    :return: built in python list with tuples: each tuple contain 2 elements: left and right window indexes
    '''
    res = [(i, i + window_length - 1) for i in range(series_len - window_length + 1)]
    return res

def get_all_adjacent_windows_indexes(series_len, window_length_lower_bound = 4, window_length_upper_bound = 14):
    '''
    Get window item indexes for windows having several sizes: between window_size_lower_bound and window_size_upper_bound
    This method facilitates the usage of get_adjacent_windows_indexes.
    :param series_len: the length of the series (sequence) for which the window is "moved" to generate subsequences
    :param window_length_lower_bound: lower bound value for the length of generated windows
    :param window_length_upper_bound: upper bound value for the length of generated windows
    :return: built in python list with tuples: each tuple contain 2 elements: left and right window indexes
    '''
    sliding_windows_indexes = []
    sliding_windows_sizes = range(window_length_lower_bound, window_length_upper_bound + 1)
    for sliding_windows_size in sliding_windows_sizes:
        adjacent_windows_indexes = get_adjacent_windows_indexes(series_len, sliding_windows_size)
        sliding_windows_indexes.extend(adjacent_windows_indexes)

    return sliding_windows_indexes


def create_sliding_windows_from_freq_and_indexes(absolute_frequencies, relative_frequencies, sliding_windows_indexes):
    '''
    Create SlidingWindowItem instances using window item delimiters indexes over the absolute frequencies and relative frequencies
    :param absolute_frequencies: np array with absolute frequencies
    :param relative_frequencies: np array with relative frequencies
    :param sliding_windows_indexes: list with tuples (sliding window indexes pairs)
    :return: built in list with SlidingWindowItem objects
    '''
    sliding_windows_objs = []
    for left_idx, right_idx in sliding_windows_indexes:
        new_sw = SlidingWindowItem(absolute_frequencies, relative_frequencies, left_idx, right_idx)
        sliding_windows_objs.append(new_sw)

    return sliding_windows_objs

def get_SlidingWindowItem_sorted_by_score(sliding_windows_objs):
    '''
    Sort the given list of SlidingWindowItem objects in descending order using score attribute as the criteria.
    The resulted list is a deep copy of the original one.
    :param sliding_windows_objs: a list of SlindingWindowItem instances
    :return: the sorted list of instances
    '''
    sliding_windows_objs_sorted = sorted(sliding_windows_objs, key = lambda sw_item : sw_item.score, reverse=True)
    return sliding_windows_objs_sorted

def create_ICSInstance_from_sliding_windows(sliding_windows_objs, sq_computation_method = "sub", ccm = "avg"):
    '''
    Create ICSInstance objects using a list of SlidingWindowItem objects: one ICSInstance for each SlidingWindowItem
    :param sliding_windows_objs: a list of SlindingWindowItem instances
    :param sq_computation_method: method used for the computation of shifting percentile
    :param ccm: method to compute centroid in ICS worker
    :return: built in list with ICSInstance objects
    '''
    ics_list = []
    for i in range(len(sliding_windows_objs)):
        new_ics = ICSEstimator(sliding_windows_objs[i].abs_freq_vals, (sliding_windows_objs[i].left_idx, sliding_windows_objs[i].right_idx),
                               sq_computation_method, ccm)
        ics_list.append(new_ics)

    return ics_list

def get_confusion_matrix(y_true, y_pred):
    '''
    Return confusion matrix created with given arrays
    :param y_true: the ground-truth values
    :param y_pred: the predicted values
    :return: the confusion matrix corresponding to the given values; it use the confusion_matrix function from sklearn
    '''
    return confusion_matrix(y_true, y_pred)

def compute_eval_metrics(conf_matrix):
    '''
    Compute several classic supervised learning evaluation metrics and return the result as a dictionary
    :param conf_matrix:
    :return:
    '''
    res = dict()
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tp = conf_matrix[1, 1]
    accuracy_val = (tp + tn) / np.sum([tp, tn ,fn, fp])
    precision_val = tp / np.sum([tp + fp])
    recall_val = tp / np.sum(tp + fn)
    specificity_val = tn / np.sum(tn + fp)
    f1_score_val = (2 * precision_val * recall_val) / (precision_val + recall_val)

    res['accuracy'] = accuracy_val
    res['precision'] = precision_val
    res['recall'] = recall_val
    res['specificity'] = specificity_val
    res['f1_score'] = f1_score_val

    return res

# OUT OF USAGE: it was used in the past for datasets with features which follow normal distributions
def create_labeled_cycles_and_ics_estimators(cycles, sw_size_lb = 4, sw_size_ub = 14, top_sliding_windows = 15):
    '''
    Auxiliary function that take unlabed cycles and assign labels to then; then create window items and ICSEstimator objects
    using the cycles labeled previously.
    :param cycles: unlabeled instances
    :param sw_size_lb: moving sliding window size, lower bound
    :param sw_size_ub: moving sliding window size, upper bound
    :param top_sliding_windows: indicate how much window items are selected to create ICSEstimator objects
    :return: a tuple with 3 values: labeled cycles, ICSEstimator objects, a repot with several information;
    every labeled cycle contain cycle values (numpy array), anomalous state of cycle and anomalous points positions
    '''
    values_by_pos = get_values_by_pos(cycles)

    # compute mean and std for every position
    pattern_len = len(cycles[0])
    mean_by_pos = np.array([compute_mean(values_by_pos[i]) for i in range(pattern_len)])
    std_by_pos = np.array([compute_std(values_by_pos[i]) for i in range(pattern_len)])

    # label cycles as anomalous and non anomalous
    labeled_cycles = label_cycles(cycles, mean_by_pos, std_by_pos, std_factor = 2, no_of_points_threshold = 7)

    report = dict()
    report['cycle_len'] = pattern_len
    report['total_cycles'] = len(cycles)
    report['anomalous_cycles'] = sum([1 for elem in labeled_cycles if elem[1] == 1])
    report['anomalous_cycles_percent'] = sum([1 for elem in labeled_cycles if elem[1] == 1]) / len(labeled_cycles)

    anomalous_points_pos_sublists = [labeled_cycle[2] for labeled_cycle in labeled_cycles if labeled_cycle[1] == 1]
    anomalous_points_pos_freq = get_anomalous_points_pos_freq(anomalous_points_pos_sublists, pattern_len)
    relative_freq = compute_relative_frequencies(anomalous_points_pos_freq)

    sw_indexes = get_all_adjacent_windows_indexes(pattern_len, sw_size_lb, sw_size_ub)
    sw_objects = create_sliding_windows_from_freq_and_indexes(anomalous_points_pos_freq, relative_freq, sw_indexes)
    sw_objects = get_SlidingWindowItem_sorted_by_score(sw_objects)
    ics_instances = create_ICSInstance_from_sliding_windows(sw_objects[0:top_sliding_windows])

    return labeled_cycles, ics_instances, report