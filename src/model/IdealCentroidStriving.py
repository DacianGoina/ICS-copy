
'''
In this first phare, this class receive already created ICSInstance objects along with train and test data.
Further, there will be an implementation in that the class receive cycles directly and perform all the steps.
'''
from src.main.cycle_utils import get_values_by_pos, get_anomalous_points_pos_freq
from src.main.statistics_utils import compute_relative_frequencies
from src.model.AnomalousBinLabeler import AnomalousBinLabeler
from src.model.model_utils import get_all_adjacent_windows_indexes, get_SlidingWindowItem_sorted_by_score, \
    create_sliding_windows_from_freq_and_indexes, create_ICSInstance_from_sliding_windows
from src.main.sampling_utils import extract_stratified_sample

import numpy as np
import random

class IdealCentroidStriving():
    def __init__(self, bins_no ="auto", anomalous_cumulated_threshold = 0.05, anomalous_points_threshold_no = 2,
                 min_sw_size = 4, max_sw_size = 14, estimators_no = 15, windows_selection_strategy ="best", sp_computation_method ="sub",
                 min_sss_p = 0.3, max_sss_p = 0.55, ss_ap = 0.02, ss_seed = None, ics_ccm = "avg"):
        '''
        :param bins_no: number of equidistant bins to be used for histograms (to estimates probability distributions);
        if "auto" then the Freedman-Diaconis rule is used, other the passed value is used
        :param anomalous_cumulated_threshold: ('score' parameter) relative frequency values of the bins are sum into a cumulated sum until this value is reached
        :param anomalous_points_threshold_no: (k paramter) minimum number of anomalous values for an instance to be considered an anomaly
        :param min_sw_size: lower bound length value of the generated window items
        :param max_sw_size: upper bound length value of the generated window items
        :param estimators_no: number of estimators to be used by the model
        :param sp_computation_method: shifting percentile computation mode (for ICS estimators); "div" or "sub"
        :param windows_selection_strategy: strategy for selection of window items used for the creation of estimators;  "best" or "random"
        :param min_sss_p: lower bound value of sampling percentage (lower bound size's value of training samples)
        :param max_sss_p: upper bound value of sampling percentage (upper bound size's value of training samples)
        :param ss_ap: (p' parameter) percentage of anomalous instaces to be selected in stratified samples used for estimators' training
        :param ss_seed: seed value used by the generator for the selection of instances for the samples used in training stage;
        if None, then a random seed value is selected by default
        :param ics_ccm: centroid computation mode for ICS estimators; "avg" (classic mean) or "wavg" (weighted average)

        Other attributes declared below:
            - X: the training data, is assigned in the fitting stage
            - ics_estimators: ICSEstimator instances; are created in the trianing stage
            - report: a dictionary containing meta information about the dataset and model; is constructed during the fitting and training stage
            - cycle_len: number of features in the dataset
        '''
        self.X = None
        self.ics_estimators = None
        self.report = dict()
        self.cycle_len = None # number of features in the dataset

        self.bins_number = bins_no
        self.anomalous_cumulated_threshold = anomalous_cumulated_threshold
        self.anomalous_points_threshold_no = anomalous_points_threshold_no
        self.min_sw_size = min_sw_size
        self.max_sw_size = max_sw_size
        self.estimators_no = estimators_no
        self.windows_selection_strategy = windows_selection_strategy
        self.sp_computation_method = sp_computation_method
        self.min_sss_p = min_sss_p
        self.max_sss_p = max_sss_p
        self.ss_ap = ss_ap
        self.ss_seed = ss_seed
        self.ics_ccm = ics_ccm

    def fit(self, X):
        '''
        Fit the model with training data.
        :param X: X data: array with cycles (1d np array of values)
        :return:
        '''

        self.X = np.array(X)
        self.cycle_len = len(self.X[0])
        self.__check_total_max_number_of_sw()


    def __get_sw_no_max_value(self):
        '''
        Compute the maximum number of window items that can be genered using using the given min_sw_size and max_sw_size
        :return: the computed value
        '''
        sw_min_size = self.min_sw_size # m
        sw_max_size = self.max_sw_size # M
        # let N be the cycle len (number of features)
        # compute sum from i = m to M of: N - i  + 1
        max_number_of_sw = sum([ (self.cycle_len - sw_size + 1) for sw_size in range(sw_min_size, sw_max_size + 1) ])
        return max_number_of_sw

    def __check_total_max_number_of_sw(self):
        '''
        Check if the given number of estimators is lower equal than the maximum number of estimators that could
        be created using the given min_sw_size and max_sw_size
        :return: if the condition described above does not hold, then an assertion error is raised and the execution stops
        '''
        max_number_of_sw = self.__get_sw_no_max_value()
        assert self.estimators_no <= max_number_of_sw , "number of used sliding windows must be <= than " + str(max_number_of_sw)


    def create_anomalous_labelers(self, anomalous_cumulated_threshold_, bins_no):
        '''
        Create AnomalousBinLabeler instances - one for each feature of the dataset.
        :param anomalous_cumulated_threshold_: anomalous_cumulated_threshold value
        :param bins_no: number or bins to be used for histogram;
        if is a numerical value, use that value, if "auto" then use Freedman-Diaconis rule to compute the bins number
        :return: built-in list with AnomalousBinLabeler objects
        '''
        values_by_pos = get_values_by_pos(self.X)
        anomalous_bin_labelers = [AnomalousBinLabeler(values_by_pos[i], anomalous_cumulated_threshold = anomalous_cumulated_threshold_, bins_number = bins_no) for i in range(len(values_by_pos))]
        return anomalous_bin_labelers

    def get_instance_anomalous_state_and_pos(self, cycle, anomalous_labelers, no_of_points_threshold):
        '''
        Determine normal / anomalous state and feature indexes of anomalous values for an instance (cycle).
        For a cycle, if it contains at least no_of_points_threshold anomalous points, then it is considered as an anomaly.
        A value is anomalous if it belongs to an anomalous bin
        :param cycle: 1d np array of values
        :param anomalous_labelers: built-in list with AnomalousBinLabeler objects
        :param no_of_points_threshold: no of points to be used when decide if a cycle is anomalous or not.
        :return: tuple with 2 values: True / False depending on whether the cycle state is anomalous or not, and a list with anomalous positions;
        if cycle is not anomalous, the list will be empty
        '''
        all_positions = range(self.cycle_len)
        anomalous_points_pos = [pos for pos in all_positions if anomalous_labelers[pos].is_value_anomalous(cycle[pos])]
        if len(anomalous_points_pos) >= no_of_points_threshold:
            return (True, anomalous_points_pos)

        return (False, anomalous_points_pos)

    def assign_artificial_labels(self, anomalous_labelers, anomalous_points_no):
        '''
        Assign artificial normal / anomalous labels for all cycles (instances) from self.X.
        For each cycle (instance), use get_instance_anomalous_state_and_pos() function to decide if it's anomalous or not.
        :param anomalous_labelers: AnomalousBinLabeler objects
        :param anomalous_points_no: number of points to consider a cycle as being anomalous (anomaly)
        :return: built-in list with tuples; each tuple contains instance's values, anomalous state (as 0 = normal, 1 = anomalous)
        and feature indexes of anomalous values for that instace
        '''

        assert anomalous_points_no < self.cycle_len, "Min anomalous points number per instance must be lower than the number of features"

        labeled_instances = [(cycle,) + self.get_instance_anomalous_state_and_pos(cycle, anomalous_labelers, anomalous_points_no) for cycle in self.X]
        labeled_instances = [(item[0], 1, item[2]) if item[1] is True else (item[0], 0, item[2]) for item in labeled_instances]
        return labeled_instances

    def compute_anomalous_points_freq_by_pos(self, labeled_instances):
        '''
        Compute absolute and relative frequencies values for anomalous positions from the artificial labeled instances.
        :param labeled_instances:  artificially labeled instances
        :return: tuple with two 1d np-array : one with abs freq, another one with relative freq
        '''
        anomalous_points_pos_sublists = [labeled_instance[2] for labeled_instance in labeled_instances if labeled_instance[1] == 1]
        anomalous_points_pos_abs_freq = get_anomalous_points_pos_freq(anomalous_points_pos_sublists, self.cycle_len)
        anomalous_points_relative_freq = compute_relative_frequencies(anomalous_points_pos_abs_freq)

        return anomalous_points_pos_abs_freq, anomalous_points_relative_freq


    def create_window_item_instances(self, min_sw_size, max_sw_size, absolute_freq, relative_freq):
        '''
        Create window item objects using absolute and relative frequencies values of feature indexes of anomalous points.
        Use a moving window approach to create all posible window items with lengths from the provided range.
        For a cycle length n (number of features) and sliding window length L, there exists n - L + 1 possible sliding windows which can be generated.
        :param min_sw_size: min size for the generated window items
        :param max_sw_size: max size for the generated window items
        :param absolute_freq: 1d np array with absolute frequencies values
        :param relative_freq: 1d np array with relative frequencies values
        :return: built-in list with SlidingWindowItem objects
        '''

        assert min_sw_size <= self.cycle_len and  max_sw_size <= self.cycle_len, \
            "min and max sliding window length values must be <= than cycle len value (number of features)"

        sw_indexes = get_all_adjacent_windows_indexes(self.cycle_len, min_sw_size, max_sw_size)
        sw_objects = create_sliding_windows_from_freq_and_indexes(absolute_freq, relative_freq, sw_indexes)

        return sw_objects

    def select_top_window_items(self, window_item_objects, N, selection_strategy):
        '''
        'best' strategy: sort window item objects in descending order using scores, then select the first N objects
        'auto' strategy: uniform selection of N window items
        :param window_item_objects: built-in list with SlidingWindowItem objects to select from
        :param N: top N window items to be selected
        :param selection_strategy: "best" or "auto"
        :return: built-in list with SlidingWindowItem objects
        '''

        assert N <= len(window_item_objects), "value of N must be <= than the number of window item objects"

        # 'best' strategy
        if selection_strategy == 'best':
            window_item_objects = get_SlidingWindowItem_sorted_by_score(window_item_objects)
            window_item_objects = window_item_objects[0:N]

        # 'auto' strategy
        else:
            proba = [1 / len(window_item_objects)] * len(window_item_objects)
            window_item_objects = random.choices(window_item_objects, weights=proba, k=N)

        return window_item_objects

    def create_ICS_estimators(self, window_item_objects, sq_computation_method ="sub"):
        '''
        Create ICSEstimator objects using the given list of window items.
        :param window_item_objects: built-in list with SlidingWindowItem objects
        :param sq_computation_method: method used for the computation of shifting percentile for estimators
        :return: built-in list with ICSEstimator objects
        '''
        ics_estimators_list = create_ICSInstance_from_sliding_windows(window_item_objects, sq_computation_method, self.ics_ccm)
        self.ics_estimators = ics_estimators_list
        return ics_estimators_list

    def perform_training(self, labeled_instances, sss_lb, sss_up, ap):
        '''
        Train the ICS estimators. Each estimator is trained separately.
        For the training of each estimator, a sample of size f is taken from artificially labeled dataset
        (select f with uniform distribution from range [sss_lb, sss_up]).
        Further, the centroid is compute for the cycles selected in the sample, then the distances are computed between
        each cycle (instance) from sample and the centroid; then the percentiles are computed

        As a remark related to the ap param's value (p' parameter): if artificially labeled dataset contains a percentage p of anomalies,
        then the maximum value for ap is (p * 1/(f)) (i.e how many anomalous cycles can be in the selected subset)
        where f is the size of selected subsample (subset) of cycles.
        As an example, if dataset contains 100k cycles (instances) and only 3k of them are anomalies (so 0.97 are normal, 0.03 are anomalous)
        and the size of the selected subset is 0.42 (i.e pick 42k cycles), then max value for ap in this case is 0.03 * (1/0.42) = 0.0714
        (approx 7%), to check this: 0.0714 * 42000 (subset cardinal) = 3000 so you can select all 3k anomalous cycles.

        :param labeled_instances: artificially labeled cycles (instances)
        :param sss_lb: lower bound size value (as percentage) of sample size
        :param sss_up: upper bound size value (as percentage) of sample size
        :param ap: anomalous cycles percentage to select in the sample, i.e for all f% cycles from sample, ap% of them
        will be anomalous cycles (anomalies); ap must be lower than the percentage of all anomalous cycles
        :return:
        '''

        labeled_instances_values = [instance_vals[0] for instance_vals in labeled_instances] # values (features)
        labeled_instances_outcomes = [instance_outcome[1] for instance_outcome in labeled_instances] # artificial labels
        for ics_estimator in self.ics_estimators:
            sample_size_val = np.round(np.random.uniform(sss_lb, sss_up), 2)
            max_allowed_anomalous_percentage = self.report['anomalous_cycles_percentage'] * (1/sample_size_val)
            assert ap < max_allowed_anomalous_percentage, \
                "The provided anomalous instances percentage {} for selection is higher than maximum allowed percentage {}".format(ap, max_allowed_anomalous_percentage)

            x_train_samples, y_train_samples = extract_stratified_sample(labeled_instances_values, labeled_instances_outcomes,
                                                    sample_size = sample_size_val,  class_one_percentage = ap, gen_seed=self.ss_seed)
            ics_estimator.fit(x_train_samples, y_train_samples)
            ics_estimator.transform()


    def update_report_multi_items(self, labeled_instances, anomalous_pos_abs_freq):
        '''
        The report includes several information related to model's creation process
        :param labeled_instances: artificially labeled cycles (instances)
        :param anomalous_pos_abs_freq: absolute frequencies for feature indexes of anomalous points from anomalous cycles
        :return:
        '''

        self.report['cycle_len'] = len(labeled_instances[0][0])
        self.report['cycles_no'] = len(labeled_instances)
        normal_cycles_no = sum([1 for cycle in labeled_instances if cycle[1] == 0])
        anomalous_cycles_no = sum([1 for cycle in labeled_instances if cycle[1] == 1])
        self.report['normal_cycles_no'] = normal_cycles_no
        self.report['anomalous_cycles_no'] = anomalous_cycles_no
        self.report['anomalous_cycles_percentage'] = anomalous_cycles_no / self.report['cycles_no']
        self.report['artificial_labels'] = [cycle[1] for cycle in labeled_instances]
        self.report['anomalous_pos_abs_freq'] = anomalous_pos_abs_freq

        self.report['min_sw_size'] = self.min_sw_size
        self.report['max_sw_size'] = self.max_sw_size
        self.report['sw_selection_strategy'] = self.windows_selection_strategy
        self.report['sliding_windows_no'] = self.estimators_no
        self.report['anomalous_cumulated_threshold'] = self.anomalous_cumulated_threshold
        self.report['bins_no'] = self.bins_number
        self.report['ss_seed'] = self.ss_seed


    def update_report(self, new_key, new_value):
        '''
        Update the report adding a new data with a given key
        :param new_key: new key to be added in report dict
        :param new_value: value to be assigned to the new added key
        :return:
        '''
        self.report[new_key] = new_value

    def transform_first_stage(self):
        '''
        Artificial labeling stage. Important: at the end of this stage, the report is usable
        :return: tuple with 3 elements: labeled_cycles (instances), anomalous_pos_abs_freq, anomalous_pos_relative_freq
        '''

        # create anomalous labelers for each position of cycles
        anomalous_labelers = self.create_anomalous_labelers(self.anomalous_cumulated_threshold, self.bins_number)

        # assign articial labels to cycles
        labeled_instances = self.assign_artificial_labels(anomalous_labelers, anomalous_points_no = self.anomalous_points_threshold_no)
        self.update_report('min_anomalous_points_per_cycle', self.anomalous_points_threshold_no)

        # get absolute and relative frequencies for anomalous points positions
        anomalous_pos_abs_freq, anomalous_pos_relative_freq = self.compute_anomalous_points_freq_by_pos(labeled_instances)

        self.update_report_multi_items(labeled_instances, anomalous_pos_abs_freq)

        return labeled_instances, anomalous_pos_abs_freq, anomalous_pos_relative_freq

    def transform_second_stage(self, labeled_instances, anomalous_pos_abs_freq, anomalous_pos_relative_freq):
        '''
        Second stage: creation of window items instances and ICS estimators; further, ICS estimators are trained
        :param labeled_instances: cycles (instances) labeled via artificial labeling
        :param anomalous_pos_abs_freq: absolute frequencies values for anomalous positions from anomalous cycles
        :param anomalous_pos_relative_freq: relative frequencies values for anomalous positions from anomalous cycles
        :return:
        '''
        # create window items objects using absolute and relative frequencies
        window_item_objects = self.create_window_item_instances(self.min_sw_size, self.max_sw_size, anomalous_pos_abs_freq, anomalous_pos_relative_freq)

        # select top sliding windows
        window_item_objects = self.select_top_window_items(window_item_objects, self.estimators_no, self.windows_selection_strategy)

        # create ICS estimators instances
        ics_estimators_list  = self.create_ICS_estimators(window_item_objects, self.sp_computation_method)

        # train ICS estimators
        self.perform_training(labeled_instances, self.min_sss_p, self.max_sss_p, self.ss_ap)

    def transform(self):
        '''
        Model's transform function that includes the both stages used in the model creation: artificial labeling stage and training stage.
        :return:
        '''
        assert self.X is not None, "No X data provided"

        labeled_cycles, anomalous_pos_abs_freq, anomalous_pos_relative_freq = self.transform_first_stage()

        self.transform_second_stage(labeled_cycles, anomalous_pos_abs_freq, anomalous_pos_relative_freq)


    def predict(self, input_instance, percentile_rank = 97):
        '''
        Model's prediction function. Each estimator computes the distance between his centroid and the given instance;
        if the obtained distance exceeds the value of the percentile corresponding to the given percentile rank, the estimator evaluates the instance
        as an anomaly (value 1), else as a normal instance (value 0). A majority votes is applied over the predicted values provided by all
        estimators to decide the final value (the most frequent value is returned)
        :param input_instance: the instance to be evaluated (an n-dimensional point, i.e the instance is represented as a lit of values)
        :param percentile_rank: the percentile rank used in the prediction
        :return: 1 if the given instance is evaluated as an anomaly, 0 otherwise
        '''
        assert self.ics_estimators is not None, "ICS estimators do not exist"

        # perform predictions using all ICS esstimators
        predicted_vals = [ics_worker.predict(input_instance, percentile_rank) for ics_worker in self.ics_estimators]

        # apply a vote system to decide the outcome
        if predicted_vals.count(1) > predicted_vals.count(0):
            return 1
        else:
            return 0

    def predicts(self, input_instances_list, percentile_rank = 97):
        '''
        Prediction function but a list of instances is passed instead of a single instance
        :param input_instances_list:
        :param percentile_rank:
        :return: list of the predicted values
        '''
        return [self.predict(input_point, percentile_rank) for input_point in input_instances_list]
