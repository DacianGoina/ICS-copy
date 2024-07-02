import sys

from src.main.cycle_utils import get_values_by_pos

sys.path.insert(0, '../main')

import numpy as np
from src.main.statistics_utils import compute_mean , compute_weighted_average_by_bins

'''
Ideal Centroid Striving Estimator (learner). An instance is created starting from absolute frequency values and a pair
of indexes (left_idx, right_idx) corresponding to the subsequence to which the given frequencies belong to
'''

class ICSEstimator():
    def __init__(self, abs_freq_vals, indexes, sp_computation_method ="sub", ccm ="avg"):
        '''

        :param abs_freq_vals: absolute frequency values (corresponding to the features delimited by indexes (a, b) from window item )
        :param indexes: features delimiters (a, b) of the given frequency values (with respect to the whole list of frequency values)
        :param sp_computation_method: shinfting percentile computation mode: "div" or "sub"
        :param ccm: centroid computation mode: "avg" or "wavg"
        Other attributes declares in this contructor:
            - alphas: weights used in the computation of the distances between centroid and training subset's instances
            - ideal: the centroid, it is computed as the normal instances from the given subset of instances (cycles)
            - cycles: the subset received in the fitting phase; the instances used for the training of the estimator
            - percentiles: the percentiles values computed for the distances computed by the estimator in the training stage
            - shifting_percentile: the rank of shifting percentile
        '''
        self.abs_freq_vals = abs_freq_vals
        self.indexes = indexes
        self.alphas = self.compute_alphas(abs_freq_vals)
        self.centroid = None
        self.cycles = None
        self.percentiles = None
        self.shifting_percentile = None
        self.sp_computation_method = sp_computation_method
        self.ccm = ccm

    def __str__(self):
        return f"ICRInstace; {self.alphas}"

    def fit(self, cycles_X, cycles_y):
        '''
        Estimator fitting phase: the cycles from the given subset are sliced computed to the corresponding indexes;
        normal instances are used to compute the centroid
        :param cycles_X: cycles
        :param cycles_y: artificial labels for cycles_X
        :return:
        '''
        sliced_cycles = np.array([cycle[self.indexes[0] : self.indexes[1] + 1] for cycle in cycles_X]) # all cycles
        x_to_y_samples = list(zip(cycles_X, cycles_y))
        normal_cycles = np.array([cycle[self.indexes[0] : self.indexes[1] + 1] for cycle, outcome in x_to_y_samples if outcome == 0]) # normal cycles

        self.cycles = sliced_cycles

        self.compute_centroid(normal_cycles)

    def compute_centroid(self, normal_cycles):
        '''
        Compute the entroid using only the normal (non anomalous cycles)
        :param normal_cycles:
        :return:
        '''

        if normal_cycles is None or len(normal_cycles) == 0:
            raise Exception("Cannot compute the centroid: the cycles are missing")

        pattern_len = len(normal_cycles[0])
        positions = range(pattern_len)
        values_by_pos = get_values_by_pos(normal_cycles)
        if self.ccm == "avg": # use classic average
            mean_by_pos = np.array([compute_mean(values_by_pos[i], remove_max = True, remove_min = True) for i in positions])
            self.centroid = mean_by_pos
        else: # use a weighted average based on bins size (number of elements) and mean of the bin edges
            weighted_average_by_pos = np.array([compute_weighted_average_by_bins(values_by_pos[i], "auto") for i in positions])
            self.centroid = weighted_average_by_pos


    def transform(self):
        '''
        Use the given trained subset (collection of cycles) to train the estimator.
        :return:
        '''
        # compute distances percentiles based on fitted values and ideal
        if self.cycles is None:
            raise Exception("Cannot train the model: cycles are missing")

        if self.centroid is None:
            raise Exception("Cannot train the model: centroid is missing")

        collected_distances = np.array([self.compute_distance(cycle) for cycle in self.cycles])
        percentiles = self.compute_percentiles(collected_distances)
        self.percentiles = percentiles

        # compute and set the shifting percentile
        self.compute_shifting_percentile(method = self.sp_computation_method)


    def compute_distance(self, input_point):
        '''
        Compute the distance between the estimator's centroid and a given instance (n-dimensional point, i.e list with n values)
        :param input_point: n-dimensional point
        :return: the computed distance
        '''
        if self.centroid is None:
            raise Exception("Cannot compute the distnace: missing ideal")

        assert len(input_point) == len(self.centroid), "input point must to have same size as ideal"

        pattern_len = len(self.centroid)
        distance_val = np.sum([self.alphas[i] * np.abs(input_point[i] - self.centroid[i]) for i in range(pattern_len)])
        return distance_val


    def predict(self, input_instance, percentile_rank = 97, print_dist = False):
        '''
        Prediction function for a single instance (i.e. row from dataset).
        :param input_instance: the instance to be evaluated (an n-dimensional point, i.e the instance is represented as a lit of values)
        :param percentile_rank: the percentile rank of the percentile compared with the computed distance
        :param print_dist: print to computed distance to the console
        :return: 1 if the given instance is evaluated as an anomaly, 0 otherwise (non anomalous)
        '''
        if self.centroid is None or self.percentiles is None:
            raise Exception("Cannot perform the predict: centroid or percentiles is missing")

        input_instance_sliced = input_instance[self.indexes[0]: self.indexes[1] + 1]

        assert input_instance_sliced is not None, "input cannot be None"
        assert len(self.centroid) == len(input_instance_sliced), "input must to have the same size as the centroid"
        if percentile_rank is None: # if None, then use shifting percentile
            percentile_rank = self.shifting_percentile
        assert type(percentile_rank) == type(1) and percentile_rank>= 1 and percentile_rank <= 99, "percentile rank must be an integer between 1 and 99"

        distance = self.compute_distance(input_instance_sliced)
        percentile_value = self.percentiles[percentile_rank]
        if print_dist is True:
            print("dist = ", distance)

        if distance >= percentile_value:
            return 1

        return 0

    def predicts(self, input_instances_list, percentile_rank = 97):
        '''
        Estimator's prediction function; a list of instances is passed instead of a single instance
        :param input_instances_list:
        :param percentile_rank:
        :return: list of the predicted values
        '''
        if self.centroid is None or self.percentiles is None:
            raise Exception("Cannot predict: missing ideal or percentiles")

        predictions = [self.predict(input_point, percentile_rank) for input_point in input_instances_list]
        return predictions


    def compute_alphas(self, abs_freq_vals):
        '''
        Compute alphas (weights) as the relative frequency values for the absolute frequency values
        :param abs_freq_vals:
        :return: alphas
        '''
        return abs_freq_vals / np.sum(abs_freq_vals)

    def compute_percentiles(self, values):
        '''
        Compute the percentiles retained by the estimator.
        :param values: the values for which the percentiles are calculated
        :return: a dict, the keys are the percentiles index / ranks (from 1 to 99) and the data are the percentiles values(numpy float64)
        '''
        percentiles_values = {i:np.percentile(values, q = i) for i in range(1, 100)}
        return percentiles_values

    def compute_shifting_percentile(self, method = "sub"):
        '''
        The shifting percentile is a percentile rank (index) such that there is a significant change / leap
        in the values between current percentile (P_q) and previous percentile (P_q-1).
        Methods to compute shifting percentile: "sub" to compute it using the difference between consecutive percentile values,
        or "div" to compute it using the result of division (report) between consecutive percentile values
        :return: the rank of the shifting percentile
        '''
        pairs = []
        q = 1
        while q <= 98:
            neighbor_elems_val = 0
            if method == "sub":
                neighbor_elems_val =  self.percentiles[q+1] - self.percentiles[q]
            else:
                neighbor_elems_val =  self.percentiles[q+1] / self.percentiles[q]

            pairs.append( (q, q+1, neighbor_elems_val) )
            q += 1

        pairs = sorted(pairs, key = lambda pair : pair[2], reverse=True)

        self.shifting_percentile = pairs[0][1]
