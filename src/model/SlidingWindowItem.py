import numpy as np

'''
Represent a window item; it consists of absolute frequency values for the feature's indexes of anomalous values (i.e. v vector),
absolute values of v (v'),  and delimiters (left_idx, right_idx) to identify the specific subsequence in v and v'
'''

class SlidingWindowItem():
    def __init__(self, freq_vals_full_vector, normalized_vals_full_vector, left_idx, right_idx):
        '''

        :param freq_vals_full_vector: (v) list with absolute frequency values (for feature's indexes of anomalous points, for whole sequence)
        :param normalized_vals_full_vector: (v') relative frequency values (for freq_vals_full_vector)
        :param left_idx: start index of window item in the sequences v and v'
        :param right_idx: end index of window item in the sequences v and v'
        Remark: right_idx is the real right_index, for slicing (e.g [a:b]) use right_idx+1
        '''

        self.abs_freq_vals = freq_vals_full_vector[left_idx:right_idx + 1]
        self.normalized_vals = normalized_vals_full_vector[left_idx:right_idx+1]
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.score = self.get_score()

    def __str__(self):
        return f"size = {self.get_window_size()}, [{self.left_idx}; {self.right_idx}]; score = {self.score}"

    def get_window_size(self):
        return self.right_idx - self.left_idx + 1

    # sum normalized_vals / window size
    def get_score(self):
        '''
        Compute the score of the window item as the sum of the values from freq_vals_full_vector
        delimited by the given indexes (left_idx, right_idx) divided to the length of the freq_vals_full_vector
        :return: the computed score
        '''
        return np.sum(self.normalized_vals) / self.get_window_size()
