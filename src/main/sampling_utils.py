from sklearn.model_selection import train_test_split
import numpy as np
import copy
import random

'''
This module includes functions used to create data samples.
'''

def split_data(x_vals, y_vals, seed_val = 4, test_size_val = 0.25):
    '''
    Split data for training and testing; use a stratified manner.
    :param x_vals:
    :param y_vals:
    :param seed_val:
    :param test_size_val:
    :return:
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, random_state=seed_val, test_size = test_size_val, stratify = y_vals)
    return x_train, x_test, y_train, y_test


def split_into_2_samples(data, size = 0.8, shuffle_seed = None):
    '''
    Split the given collection (list) of data into two samples: first sample will contain %size of all items from the given list :param data
    and the second sample will contain the rest of (1-size)% items.
    :param data: data to select from
    :param size: percentage size for first sample
    :param shuffle_seed: shuffle_seed (for deterministic usage if needed)
    :return:
    '''
    data = copy.deepcopy(data)

    # compute split index
    split_index = int(len(data) * size)

    # shuffle
    random.Random(shuffle_seed).shuffle(data)

    # split into 2 samples
    first_sample = data[0:split_index]
    second_sample = data[split_index:]

    return first_sample, second_sample

def extract_stratified_sample(x_vals, y_vals, sample_size = 0.3,  class_one_percentage = 0.02, gen_seed = None):
    '''
    Select a sample in a stratified manner. E.g: from a given collection of arrays x_vals and a array of labels y_vals,
    take a sample of sample_size percentage (e.g 30%) in a stratified manner: ensure that for the selected items  30% from all x_vals),
    0.02% of elements belong to class 1 (anomalous),  and the rest of 0.98 belong to class 0
    :param x_vals: X values (list of arrays)
    :param y_vals: y values (artificial labels for instances from x_vals)
    :param sample_size: percentage of cycles (instances) to select in the sample
    :param class_one_percentage: percentage of anomalous instances to be selected in the sample
    :param gen_seed: generator seed value (for deterministic usage if needed)
    :return: tuple with 2 elements: selected cycles (instances) and corresponding outcomes
    '''
    if gen_seed is not None:
        np.random.seed(gen_seed)

    y_vals = np.array(y_vals)

    x_vals = np.array(x_vals)

    # get indices of class 1 and class 0 elements
    class_1_indices = np.where(y_vals == 1)[0]
    class_0_indices = np.where(y_vals == 0)[0]

    # compute the number of samples needed from each class
    class_1_samples_size = int(class_one_percentage * sample_size * len(x_vals))  #   e.g class_one_percentage of 30% of total samples
    class_0_samples_size  = int((1 - class_one_percentage) * sample_size * len(x_vals))  # e.g (1 - class_one_percentage) of 30% of total samples

    # randomly choose indices for sample items for each class
    class_1_samples_indices = np.random.choice(class_1_indices, size=class_1_samples_size, replace=False)
    class_0_samples_indices = np.random.choice(class_0_indices, size=class_0_samples_size, replace=False)

    # concatenate the chosen indices
    sampled_indices = np.concatenate([class_1_samples_indices, class_0_samples_indices])

    # get the corresponding rows from A_x and A_y
    x_selected_vals = x_vals[sampled_indices]
    y_selected_vals = y_vals[sampled_indices]

    return x_selected_vals, y_selected_vals
