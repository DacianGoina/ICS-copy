'''
Use to label values or anomalous or not using a histogram - bins strategy.
Initial list of values is split into equidist bins; the number of bins can by given manually or can be computed
using Freedman-Diaconis rule.

One bin can be characterized by bin index, bin size, score and edges
- bin index: just a numerical index used to refer the bin; index start from 1 (see observation below)
- bin size (count):  how many values are in  bin
- bin score: computed as relative frequency with respect to all the bins;
- bin edges: lower and upper bounds for values inside the bin
Observation: to create N bins, N+1 edges are required, e.g with values a0, a1, a2, a3 we can create 3 bins:
b1 with bounds [a0, a1), b2 with bounds [a1, a2), b3 with bounds [a2, a3].

After creating the bins, each bin has an index, size (count) and edges; then, scores are computed using the above mention method.
Further, bins are sorted  in ascending order using the scores. The indexes of weak / anomalous bins are collected:
a bin is considered as being weak / anomalous if it is in the group of bins whose score sum up to anomalous_cumulated_threshold.
Additional explination: after sorting the bins depending on the score, bins with low values
(i.e bins with low count - small number of values inside) are in the first places; starting with small bins,
bins are collected in a group and bins scores are added to a cumulated sum until a threshold is reached (e.g 0.15);
so bins from this group contain values with low frequencies in whole histogram, i.e these values have low changes (probabilities) to occur,
thus we consider them an anomalous values.

When a new value come, we determine it's bin (where it fit) and if the determined bin is an anomalous one, then the value is anomalous

'''

import numpy as np
import copy

from src.main.algorithms_utils import check_value_in_list
from src.main.statistics_utils import create_bins


class AnomalousBinLabeler():

    def __init__(self, input_values, anomalous_cumulated_threshold = 0.1, bins_number = "auto"):
        '''
        :param input_values:
        :param anomalous_cumulated_threshold:
        :param bins_number: number of equidistant bins to be used for split; if value is 'auto' then a suitable number
        is computed using Freedman-Diaconis rule; see numpy doc
        '''
        bins, bins_edges = self.__create_bins(input_values, bins_number)
        bin_index_to_proba = self.__compute_bins_proba(bins, bins_edges)
        anomalous_bins = self.__collect_anomalous_bins_indexes(bin_index_to_proba, anomalous_cumulated_threshold)

        self.bin_edges = bins_edges
        self.bin_index_to_proba = bin_index_to_proba
        self.anomalous_bins = anomalous_bins


    def __create_bins(self, input_values, bins_number):
        # return bins, bin_edges
        return create_bins(input_values, bins_number)

    def __compute_bins_proba(self, bins, bins_edges):
        '''
        Compute relative bin counts values - these values are used in the selection of anomalous bins; retain bin index, score and edges
        :param bins:
        :param bins_edges:
        :return:
        '''
        proba = bins / np.sum(bins)
        bin_index_to_proba = [(index+1, proba, (bins_edges[index], bins_edges[index+1] ) ) for index, proba in enumerate(proba)]

        return bin_index_to_proba

    def __collect_anomalous_bins_indexes(self, bin_index_to_proba, anomalous_cumulated_threshold):
        '''
        Select anomalous bins and return their indexes.
        :param bin_index_to_proba:
        :param anomalous_cumulated_threshold: the relative count values of the bins are added into a cumulative sum until this
        threshold value is reached
        :return: list with indexes of anomalous bins (with respect to the list with indexes of all bins)
        '''
        bin_index_to_proba = sorted(copy.deepcopy(bin_index_to_proba), key=lambda x: x[1])
        actual_score = np.float64(0.0)
        collected_bins_indexes = []

        i = 0
        while actual_score < anomalous_cumulated_threshold:
            actual_score = actual_score + bin_index_to_proba[i][1]
            collected_bins_indexes.append(bin_index_to_proba[i][0])
            i = i + 1

        collected_bins_indexes.sort()
        return collected_bins_indexes

    def __str__(self):
        return "AnomalousBinLabeler; no. of bins: " + str(len(self.bin_edges) -1)

    def __determine_suitable_bin(self, value):
        '''
        np.digitize determines the appropiate bin for placing a given value, using bin edges as criteria;
        output index value i refer to the index of the right edge of the suitable bin;
        so the given value belongs to the bin defined by edges indexes i-1 and i
        :param value: value to be evaluated
        :return: place of the given value in a specific bin, using bin edges
        '''

        bin_right_edge_index = np.digitize(np.array([value]), self.bin_edges)[0]
        #print(self.bin_index_to_proba[bin_right_edge_index - 1])
        return bin_right_edge_index


    def is_value_anomalous(self, value):
        '''
        Check if the given values is anomalous or not. A value is considered as being anomalous (anomaly) if it belongs to an anomalous bin.
        :param value:
        :return: True if the value is evaluated as anomalous, False otherwise
        '''
        bin_index = self.__determine_suitable_bin(value)

        if check_value_in_list(bin_index, self.anomalous_bins) is True:
            return True

        return False




