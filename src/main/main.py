from src.model.IdealCentroidStriving import IdealCentroidStriving
from src.main.cycle_utils import extract_non_overlayed_cycles_indexes, extract_cycles_from_ts, scale_df_data
from src.main.io_utils import import_csv_as_df, import_pkl_obj
from src.main.sampling_utils import split_into_2_samples
from matching_utils import *


if __name__ == "__main__":
    # Here is a demo for the usage of ICS and the comparison (matching) of predictions provided by ICS with prediction provided by IF.

    # import data
    df1 = import_csv_as_df("../../data/csv/big/B827EB019C33_sid_1.csv")
    data1 = scale_df_data(df1)

    # extract cycles (instances)
    matches = import_pkl_obj("../../data/binary/B827EB019C33_sid_1_cycles_1dot5std.pkl")
    cycles_indexes = [pair[1] for pair in matches]
    cycles_indexes = extract_non_overlayed_cycles_indexes(cycles_indexes, pattern_len=24)
    cycles = extract_cycles_from_ts(cycles_indexes, 24, data1)
    cycles = cycles[0:20000]

    x_train, x_test = split_into_2_samples(cycles, size= 0.9, shuffle_seed=10)

    # train ICS model
    ics = IdealCentroidStriving(windows_selection_strategy="best", estimators_no= 15,
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
    l1_types = (0,1 )
    l2_types = (1, -1)
    results =  two_way_matching(ics_labels, if_labels, l1_types, l2_types)
    matching_report = two_way_matching_report(count_labels_by_types(ics_labels, l1_types), count_labels_by_types(if_labels, l2_types) ,results)

    # print the matching report
    print(matching_report)