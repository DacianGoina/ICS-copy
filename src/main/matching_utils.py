
'''
This module contains methods that faclilitate the usage of specific unsupervised classifiers.
The results provided by specific classifiers are compared with the results provided by ICS to check the matching between
the results (predicted outcomes for instances evaluated as normal / anomalous). This is done in order to measure the quality of
a unsupervised learning model (this approach is used considered that in unsupervised learning tasks, ground truth labels aren't provided,
thus metrics such as accuracy, recall etc cannot be computed).

The functions that execute specific classifiers (e.g Isolation Forest) are trained on training data provided as parameter and
tested on testing data provided also as parameters; the result consists on predicted labels for the testing instances.

Remark related to the labels' meaning: in the case of ICS 0 means normal and 1 means anomaly,
but for the most of sklearn unsupervised learning models, 1 means normal and -1 means anomaly.

In order to check the quality of prediction, a matching (intersection) strategy is used: for 2 list of labels, provided by 2
distinct methods, the values are compared for each pair of labels: a predicted value from the first list
with his corresponding predicted value from second list (check for equality between these values).
'''

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
import copy


def execute_isolation_forest(x_train, x_test, contamination_value  = 0.03, n_estimators_val = 100):
    # Isolation Forest https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    # 1: normal, -1: anomaly
    x_train = copy.deepcopy(x_train)
    x_test = copy.deepcopy(x_test)

    isolation_model = IsolationForest(contamination=contamination_value, n_estimators=n_estimators_val)
    isolation_model.fit(x_train)
    labels = isolation_model.predict(x_test)
    labels = list(labels)
    return labels

def execute_one_class_svm(x_train, x_test, kernel_val = 'rbf', max_iter_val = 10):
    # One Class SVM https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    # 1: normal, -1: anomaly
    x_train = copy.deepcopy(x_train)
    x_test = copy.deepcopy(x_test)

    one_class_svm = OneClassSVM(kernel=kernel_val, max_iter=max_iter_val)
    one_class_svm.fit(x_train)
    labels = one_class_svm.predict(x_test)
    labels = list(labels)
    return labels

def execute_LOF(x_train, x_test, n_neighbors_val = 20):
    # Local Outlier Factor https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    # 1: normal, -1: anomaly
    x_train = copy.deepcopy(x_train)
    x_test = copy.deepcopy(x_test)

    lof_clf = LocalOutlierFactor(n_neighbors=n_neighbors_val, novelty = True)
    lof_clf.fit(x_train)
    labels = lof_clf.predict(x_test)
    labels = list(labels)
    return labels

def execute_SGDOneClassSVM(x_train, x_test, nu_val = 0.5, max_iter_val = 1000):
    # SGD OC SVM https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html
    # 1: normal, -1: anomaly
    x_train = copy.deepcopy(x_train)
    x_test = copy.deepcopy(x_test)

    sgd_oc_svm = SGDOneClassSVM(nu = nu_val, max_iter= max_iter_val)
    sgd_oc_svm.fit(x_train)
    labels = sgd_oc_svm.predict(x_test)
    labels = list(labels)
    return labels

def two_way_matching(m1_res_labels, m2_res_labels, m1_types, m2_types):
    '''
    Two way matching (intersection) between the labels predicted by 2 distinct classification methods.
    :param m1_res_labels: predicted labels provided by the first method; built-in list
    :param m2_res_labels: predicted labels provided by the second method; built-in list
    :param m1_types: label values meaning for labels from the first list: tuple with 2 values, first value refers to normal label,
    second value refers to anomalous label
    :param m2_types: label values meaning for labels from the second list: tuple with 2 values, same meaning as above
    :return: tuple with 3 numeric values: no. of matches for normal instances, no. of matches for anomalous instances, and wrong matches
    '''
    m1_normal_label, m1_anomalous_label = m1_types
    m2_normal_label, m2_anomalous_label = m2_types

    normal_instances_matches = 0
    anomalous_instances_matches = 0
    wrong_matches = 0
    for l1_label, l2_label in zip(m1_res_labels, m2_res_labels):
        if l1_label == m1_normal_label and l2_label == m2_normal_label:
            normal_instances_matches += 1
        elif l1_label == m1_anomalous_label and l2_label == m2_anomalous_label:
            anomalous_instances_matches += 1
        else:
            wrong_matches +=1

    return (normal_instances_matches, anomalous_instances_matches, wrong_matches)

def three_way_matching(m1_res_labels, m2_res_labels, m3_res_labels, m1_types, m2_types, m3_types):
    '''
    Three way matching (intersection) between the predicted labels provided by 3 distinct classification methods.
    Parameters and return meanings are the same as in the case of two_way_matching, but adapted for 3 way matching.
    '''
    m1_normal_label, m1_anomalous_label = m1_types
    m2_normal_label, m2_anomalous_label = m2_types
    m3_normal_label, m3_anomalous_label = m3_types

    normal_instances_matches = 0
    anomalous_instances_matches = 0
    wrong_matches = 0
    for l1_label, l2_label, l3_label in zip(m1_res_labels, m2_res_labels, m3_res_labels):
        if l1_label == m1_normal_label and l2_label == m2_normal_label and l3_label == m3_normal_label:
            normal_instances_matches += 1
        elif l1_label == m1_anomalous_label and l2_label == m2_anomalous_label and l3_label == m3_anomalous_label:
            anomalous_instances_matches += 1
        else:
            wrong_matches +=1

    return (normal_instances_matches, anomalous_instances_matches, wrong_matches)

def four_way_matching(m1_res_labels, m2_res_labels, m3_res_labels, m4_res_labels, m1_types, m2_types, m3_types, m4_types):
    m1_normal_label, m1_anomalous_label = m1_types
    m2_normal_label, m2_anomalous_label = m2_types
    m3_normal_label, m3_anomalous_label = m3_types
    m4_normal_label, m4_anomalous_label = m4_types

    normal_instances_matches = 0
    anomalous_instances_matches = 0
    wrong_matches = 0
    for l1_label, l2_label, l3_label, l4_label in zip(m1_res_labels, m2_res_labels, m3_res_labels, m4_res_labels):
        if l1_label == m1_normal_label and l2_label == m2_normal_label and l3_label == m3_normal_label and l4_label == m4_normal_label:
            normal_instances_matches += 1
        elif l1_label == m1_anomalous_label and l2_label == m2_anomalous_label and l3_label == m3_anomalous_label and l4_label == m4_anomalous_label:
            anomalous_instances_matches += 1
        else:
            wrong_matches +=1

    return (normal_instances_matches, anomalous_instances_matches, wrong_matches)

def count_labels_by_types(labels, types):
    '''
    Compute the frequency for each of two unique values from the given list :param labels.
    :param labels: list with labels
    :param types: tuple with unique label values: first value correspond to normal instaces, the second values with anomalous instances
    :return: tuple with 2 values: frequency (count) of occurrence of normal instances and
    frequency (count) of occurrence of anomalous instances
    '''
    normal_label, abnormal_label = types
    count_normal = sum([1 for label in labels if label == normal_label])
    count_anomalous = len(labels) - count_normal

    return (count_normal, count_anomalous)

def two_way_matching_report(m1_situation, m2_situation, matching_situation):
    '''
    A report with information related to matching (intersection) between labels provided by 2 distinct classification methods.
    :param m1_situation: normal and anomalous instances labels frequencies (raw counters) values provided by the first method
    :param m2_situation: normal and anomalous instances labels frequencies (raw counters) values provided by the second method
    :param matching_situation: tuple with results provided by a matching method (e.g two way matching)
    :return: a dictionary containing some statistic information, keys names are very suggestive
    '''
    report = dict()

    m1_count_normal, m1_count_anomalous = m1_situation
    m2_count_normal, m2_count_anomalous = m2_situation
    normal_instances_matches, anomalous_instances_matches, wrong_matches = matching_situation

    report['instances'] = m1_count_normal + m1_count_anomalous
    report['m1_normal_instances'] = m1_count_normal
    report['m1_anomalous_instances'] = m1_count_anomalous

    report['m2_normal_instances'] = m2_count_normal
    report['m2_anomalous_instances'] = m2_count_anomalous

    report['normal_instances_matches'] = normal_instances_matches
    report['anomalous_instances_matches'] = anomalous_instances_matches
    report['wrong_matches'] = wrong_matches

    return report

def three_way_matching_report(m1_situation, m2_situation, m3_situation, matching_situation):
    '''
    A report with information related to matching (intersection) between labels provided by 3 distinct classification methods.
    Similar to two_way_matching_report but adapted to three matching case.
    '''
    report = dict()

    m1_count_normal, m1_count_anomalous = m1_situation
    m2_count_normal, m2_count_anomalous = m2_situation
    m3_count_normal, m3_count_anomalous = m3_situation
    normal_instances_matches, anomalous_instances_matches, wrong_matches = matching_situation

    report['instances'] = m1_count_normal + m1_count_anomalous
    report['m1_normal_instances'] = m1_count_normal
    report['m1_anomalous_instances'] = m1_count_anomalous

    report['m2_normal_instances'] = m2_count_normal
    report['m2_anomalous_instances'] = m2_count_anomalous

    report['m3_normal_instances'] = m3_count_normal
    report['m3_anomalous_instances'] = m3_count_anomalous

    report['normal_instances_matches'] = normal_instances_matches
    report['anomalous_instances_matches'] = anomalous_instances_matches
    report['wrong_matches'] = wrong_matches

    return report

def four_way_matching_report(m1_situation, m2_situation, m3_situation, m4_situation, matching_situation):
    '''
    A report with information related to matching (intersection) between labels provided by 3 distinct classification methods.
    Similar to two_way_matching_report but adapted to four matching case.
    '''
    report = dict()

    m1_count_normal, m1_count_anomalous = m1_situation
    m2_count_normal, m2_count_anomalous = m2_situation
    m3_count_normal, m3_count_anomalous = m3_situation
    m4_count_normal, m4_count_anomalous = m4_situation
    normal_instances_matches, anomalous_instances_matches, wrong_matches = matching_situation

    report['instances'] = m1_count_normal + m1_count_anomalous
    report['m1_normal_instances'] = m1_count_normal
    report['m1_anomalous_instances'] = m1_count_anomalous

    report['m2_normal_instances'] = m2_count_normal
    report['m2_anomalous_instances'] = m2_count_anomalous

    report['m3_normal_instances'] = m3_count_normal
    report['m3_anomalous_instances'] = m3_count_anomalous

    report['m4_normal_instances'] = m4_count_normal
    report['m4_anomalous_instances'] = m4_count_anomalous

    report['normal_instances_matches'] = normal_instances_matches
    report['anomalous_instances_matches'] = anomalous_instances_matches
    report['wrong_matches'] = wrong_matches

    return report