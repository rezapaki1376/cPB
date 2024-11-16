# cPB: Continuous Piggyback Architecture for Evolving Streaming Time Series
This repository contains the code used for the experimentation shown in the paper presented at the 33rd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning Conference.


## 1) Installation
execute:

`conda create -n cPB python=3.11`

`conda activate cPB`

`pip install -r requirements.txt`

#### datasets

# SINE
It contains the generated data streams.
Testing Datasets:
* **sine_rw10_mode5_extended_16-16_1234**: S1+, S2+, S1-, S2-.
* **sine_rw10_mode5_extended_16-16_1432**: S1+, S2-, S1-, S2+.
* **sine_rw10_mode5_extended_16-16_2143**: S2+, S1+, S2-, S1-.
* **sine_rw10_mode5_extended_16-16_2341**: S2+, S1-, S2-, S1+.

pretraining used Datasets:
* **sine_rw10_mode5_extended_6-6_1234**: S1+, S2+, S1-, S2-.
* **sine_rw10_mode5_extended_6-6_1432**: S1+, S2-, S1-, S2+.
* **sine_rw10_mode5_extended_6-6_2143**: S2+, S1+, S2-, S1-.
* **sine_rw10_mode5_extended_6-6_2341**: S2+, S1-, S2-, S1+.

# Weather

Testing Datasets:
* **weather_st124_1conf**: W1+, W2+, W1-, W2-.
* **weather_st124_2conf**: W1+, W2-, W1-, W2+.
* **weather_st124_3conf**: W2+, W1+, W2-, W1-.
* **weather_st124_4conf**: w2+, W1-, W2-, W1+.

pretraining used Datasets:
* **weather_pretraining**: W1

#### utils
It contains all the pre-required functions such as preprocessing, metrics calculation, etc.

#### models
It contains the python modules implementing cPB and cGRU and the architecture of PB.

#### run-test
This folder contains the Jupyter notebooks for running pre-trained models and executing the models.

## Credits
[https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing](https://github.com/federicogiannini13/cpnn)

[https://github.com/gziffer/tenet?tab=readme-ov-file](https://github.com/gziffer/tenet?tab=readme-ov-file)
