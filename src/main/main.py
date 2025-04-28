from src.model.IdealCentroidStriving import IdealCentroidStriving
from src.main.sampling_utils import split_into_2_samples
from matching_utils import *


if __name__ == "__main__":
    # Here is a demo for the usage of ICS and the comparison (matching) of predictions provided by ICS with prediction provided by IF.

    # import data
    df = pd.read_csv("../../data/data-file.csv")

    data_as_values = df.values.tolist()

    # take 90% of data for training, 10% of data for testing (prediction)
    x_train, x_test = split_into_2_samples(data_as_values, size= 0.9, shuffle_seed=10)

    # train ICS model
    ics = IdealCentroidStriving(windows_selection_strategy="best", estimators_no= 15, min_sw_size = 4, max_sw_size = 14,
                                anomalous_points_threshold_no=10, sp_computation_method = "sub", ics_ccm = "avg", ss_ap=0.03, ss_seed=15)
    ics.fit(x_train)
    ics.transform()
    print(ics.report['anomalous_cycles_percentage'])
    print(len(ics.ics_estimators))
    for key_v, data_v in ics.report.items():
        print(key_v, " : ", data_v)

    # make prediction and compare the prediction results with the results provided by IF
    ics_labels = ics.predicts(x_test, percentile_rank=98)
    if_labels = execute_isolation_forest(x_train, x_test, contamination_value=0.038, n_estimators_val=120)
    l1_types = (0,1 ) # in ICS implementation, output value 1 means anomalous sample, value 0 means non-anomalous sample
    l2_types = (1, -1) # in sklean implementation for IF, output value -1 means anomalous sample, value 1 means non-anomalous sample
    results =  two_way_matching(ics_labels, if_labels, l1_types, l2_types)
    matching_report = two_way_matching_report(count_labels_by_types(ics_labels, l1_types), count_labels_by_types(if_labels, l2_types) ,results)

    # print the matching report
    print(matching_report)