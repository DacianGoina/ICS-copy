import numpy as np
import copy
from scipy.stats import kstest
from scipy import stats

from src.main.plot_utils import plot_hist

'''
This module contains auxiliary functions for statistical-related procedures.
'''

def compute_mean(array_obj, remove_max = False, remove_min = False):
    '''
    Return the mean value of the elements from the given numpy array.
    :param array_obj:
    :param remove_max:
    :param remove_min:
    :return:
    :rtype: numpy.float64
    '''
    array_obj_copy = copy.deepcopy(array_obj)

    max_val = np.max(array_obj_copy)
    min_val = np.min(array_obj_copy)

    # remove max elements
    if remove_max is True:
        mask_max = array_obj_copy != max_val
        array_obj_copy = array_obj_copy[mask_max]


    # remove min elements
    if remove_min is True:
        mask_min = array_obj_copy != min_val
        array_obj_copy = array_obj_copy[mask_min]

    return np.mean(array_obj_copy)


def compute_std(array_obj, remove_max = False, remove_min = False):
    '''
    Return the standard deviation value of the elements from the given numpy array.
    :param array_obj:
    :param remove_max:
    :param remove_min:
    :return:
    :rtype: numpy.float64
    '''
    array_obj_copy = copy.deepcopy(array_obj)

    max_val = np.max(array_obj_copy)
    min_val = np.min(array_obj_copy)

    # remove max elements
    if remove_max is True:
        mask_max = array_obj_copy != max_val
        array_obj_copy = array_obj_copy[mask_max]


    # remove min elements
    if remove_min is True:
        mask_min = array_obj_copy != min_val
        array_obj_copy = array_obj_copy[mask_min]

    return np.std(array_obj_copy)

def is_gaussian_distribution(values_by_pos, show_plot = False, remove_min = True, remove_max = True):
    '''
    Check if given values are drawn from a gaussian (normal) distribution
    :param values_by_pos:
    :param show_plot:
    :param remove_min:
    :param remove_max:
    :return:
    '''
    values_by_pos_copy = copy.deepcopy(values_by_pos)

    # get max and min values
    max_val = np.max(values_by_pos_copy)
    min_val = np.min(values_by_pos_copy)

    # remove max elements
    if remove_max is True:
        mask_max = values_by_pos_copy != max_val
        values_by_pos_copy = values_by_pos_copy[mask_max]
        #print("after removing max: ", len(values_by_pos_copy))


    # remove min elements
    if remove_min is True:
        mask_min = values_by_pos_copy != min_val
        values_by_pos_copy = values_by_pos_copy[mask_min]
        #print("after removing min: ", len(values_by_pos_copy))

    if show_plot:
        plot_hist(values_by_pos_copy)

    stat, p = kstest(values_by_pos_copy, stats.norm.cdf)

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # Interpret the results
    alpha = 0.05
    if p < alpha:
        print('Sample looks normal; accept Ha')
        return True
    else:
        print('Sample does not look normal: accept H0')
        return False

def compute_relative_frequencies(a):
    '''
    Compute relative frequencies for elements from array a.
    :param a: numpy 1d array
    :return: numpy 1d array with relative frequencies
    '''
    array_obj_copy = copy.deepcopy(a)
    sum_of_elements = np.sum(array_obj_copy)
    relative_freqs = array_obj_copy / sum_of_elements
    return relative_freqs

def create_bins(input_values, bins_number):
    '''
    Create bins using the given values; assume that the given values belong to the range [0, 1]
    :param input_values: input values (built in list or numpy 1d array)
    :param bins_number: number of bins; for "auto" the number is computed using Freedman-Diaconis rule (see numpy doc).
    :return: bins with bins counts (i.e number of elements in bin), and bin edges; OBS: len(bins) = N implies len(bin_edges) = N + 1
    '''
    bins, bin_edges = np.histogram(input_values, bins=bins_number)

    # check bin edges to include boundary from 0 to 1
    if bin_edges[0] > 0 :
        bin_edges = np.insert(bin_edges, 0, np.float64(0.0), axis=0)
        bins = np.insert(bins, 0, np.int64(0))


    if bin_edges[len(bin_edges) - 1] < 1:
        bin_edges = np.append(bin_edges, np.float64(1.0))
        bins = np.append(bins, [0], np.int64(0))

    return bins, bin_edges

def compute_weighted_average_by_bins(input_values, bins_no = "auto"):
    '''
    Compute the weighted average for given values using a histogram based approach. Split values into bins, then compute relative count values for
    each bin. Each component (term) for the weighted average value is computed using the specific bin: compute mean between left and right edges
    of bin then multiply it by bin relative count value (probability).
    E.g for bins edges a0, a1, a2, a3 and bins sizes b1, b2, b3, value for first bin: v1 = ((a0 + a1) / 2) * proba(b1): repeat this
    for each bin and sum all obtained values.
    :param input_values: input values
    :param bins_no: number of bins
    :return: weighted average of given elements
    '''
    bins, bin_edges = create_bins(input_values, bins_no)
    bins_proba = compute_relative_frequencies(bins)
    weighted_average = np.sum([ ( (bin_edges[i] + bin_edges[i+1]) / 2) * bins_proba[i] for i in range(len(bin_edges) - 1) ])
    return weighted_average