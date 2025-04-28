# Ideal Centroid Striving: An Unsupervised and Prediction Parameterized Anomaly Detection Method

This repository contains Python implementation for the Ideal Centroid Striving classification method presented in the paper *Ideal Centroid Striving: An Unsupervised and Prediction Parameterized Anomaly Detection Method* (ref. below).

## Usage example
The following code snippets are examples of instantiating, training, and using the model. These code snippets are from *src/main/main.py*.

**Import data**
```
# a tabular dataset with 110,000 samples (rows) and 24 features (columns)
df = pd.read_csv("../../data/data-file.csv")

data_as_values = df.values.tolist()

# take 90% of data for training, 10% of data for testing (prediction)
x_train, x_test = split_into_2_samples(data_as_values, size= 0.9, shuffle_seed=10)
```

**Instantiate an ICS model**
```
# instantiate a model such that:
	- the model uses 15 estimators
	- each estimator is trained on a minimum of 4 features and a maximum of 14 features
	- a sample from dataset is considered as being anomalous if it contains anomalous values in at least 10 features (from all 24 features)
	- 3% of (artifically-labeled) anomalous samples are used in for the training 

ics_model = IdealCentroidStriving(min_sw_size = 4, max_sw_size = 14, 
                                  estimators_no = 15,  anomalous_points_threshold_no = 10,
								  ss_ap = 0.03)
```

**Feed the model instance with data and perform training**
```
ics_model.fit(x_train)
ics_model.transform()
```

**Usage of prediction function**
```
# as anomalous score threshold use value of the 98th percentile (obtained from trained data) 

ics_labels = ics_model.predicts(x_test, percentile_rank=98)
print(ics_labels[0:10])
```

## Access the paper
The copyright-free PDF of the paper is available for free on ResearchGate - ([(PDF) Ideal Centroid Striving: An Unsupervised and Prediction Parameterized Anomaly Detection Method](https://www.researchgate.net/publication/389388217_Ideal_Centroid_Striving_An_Unsupervised_and_Prediction_Parameterized_Anomaly_Detection_Method)).

## Citation
If you use code from this repository or refer the paper, you can use BibTex citation: 
```
@INPROCEEDINGS{10896402,
  author={Goina, Dacian},
  booktitle={2024 26th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC)}, 
  title={Ideal Centroid Striving: An Unsupervised and Prediction Parameterized Anomaly Detection Method}, 
  year={2024},
  volume={},
  number={},
  pages={173-181},
  doi={10.1109/SYNASC65383.2024.00039}
 }
```