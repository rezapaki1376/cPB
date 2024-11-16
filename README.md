# cPB: Continuous Piggyback Architecture for Evolving Streaming Time Series
This repository contains the code used for the experimentation shown in the paper presented at the 33rd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning Conference.


## 1) Installation
execute:

`conda create -n cPB python=3.11`

`conda activate cPB`

`pip install -r requirements.txt`

## 2) Project structure


# Dataset Description

This repository includes synthetic and real-world datasets designed for evaluating temporal dependencies and contextual complexities in machine learning models. Below is a comprehensive description of the datasets used.

---

## SINE Data Stream

The **SINE** data generator creates two-dimensional data points for classification tasks using a sine function as its core feature. Originally, it lacked temporal dependencies. However, in this repository, we use the **SINE RW Mode**, an advanced version of SINE, which incorporates temporal dependencies within both features and labels. This enhances its suitability for machine learning models designed to handle sequential and interdependent data.

### Feature Generation
SINE RW starts with an initial data point within the range (0,1). Each subsequent point is generated through a random walk process:
- A random value, bounded by ±0.05, is added to each feature.
- The direction of the change is randomly determined.
- If a value exceeds the range (0,1), the direction is reversed to keep the data within bounds.

### Boundary Functions
The SINE RW Mode defines class boundaries using sinusoidal functions. The primary boundary functions are:

$$
S_1: x_1 - \sin(x_2) = 0
$$

$$
S_2: x_1 - 0.5 - 0.3 \sin(3 \pi x_2) = 0
$$

These boundaries yield four classification functions:
- \(S_{1+}\) and \(S_{2+}\): Assign a label "1" to points above the curve and "0" otherwise.
- \(S_{1-}\) and \(S_{2-}\): Invert the labels of \(S_{1+}\) and \(S_{2+}\).

### Temporal Label Dependency
In the SINE RW Mode, temporal dependency is introduced into the labels. Each label \(y'_t\) is determined based on the most frequent label from the last four time steps:


$$
y'_t = MODE(y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4})
$$


### Concept Drift
The data stream transitions between boundary functions (\(S_1\) and \(S_2\)) to simulate mild and severe concept drifts. For example:
- Mild drift: \(S_{1+} \to S_{2+}\)
- Severe drift: \(S_{2+} \to S_{1-}\)

By varying parameters \(\alpha\), \(\beta\), and \(\gamma\), new boundary functions can be created to test model adaptability to unique temporal patterns.

---

## Weather Data Stream

This dataset, provided by the Agricultural Research Service of the U.S. Department of Agriculture, offers real-world data ideal for studying temporal dependencies. It captures hydrometeorological variables recorded hourly from 2004 to 2014 in the Johnston Draw watershed, Idaho, USA. This region spans 1.8 square kilometers with a 372-meter elevation gradient.

### Features
The dataset includes the following features:
- Air temperature (\(T_a\)) in °C.
- Relative humidity (\(RH\)) in percentage.
- Wind speed (\(w_s\)) in m/s.
- Wind direction (\(w_d\)) in degrees (0°–360°).
- Dew point temperature (\(T_d\)) in °C.

### Preprocessing
- Features are standardized using a Robust Scaler to ensure comparability.
- Temporal dependencies are embedded by defining binary classification functions based on \(T_a\).

### Classification Functions
The dataset uses three sets of boundary functions:

**Short-Term Dynamics**:
$$
W_{1+}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) \geq T_a(t-1) \\
0, & \text{otherwise}
\end{cases}
$$

$$
W_{1-}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) < T_a(t-1) \\
0, & \text{otherwise}
\end{cases}
$$

**Extended Temporal Dependencies**:
$$
W_{2+}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) \geq \text{Median}(T_a(t-10), \dots, T_a(t-1)) \\
0, & \text{otherwise}
\end{cases}
$$

$$
W_{2-}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) < \text{Median}(T_a(t-10), \dots, T_a(t-1)) \\
0, & \text{otherwise}
\end{cases}
$$

**Pre-Training Function**:
$$
W_{3+}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) \geq \text{Min}(T_a(t-10), \dots, T_a(t-1)) \\
0, & \text{otherwise}
\end{cases}
$$

$$
W_{3-}: y(X_t) = 
\begin{cases} 
1, & \text{if } T_a(t) < \text{Min}(T_a(t-10), \dots, T_a(t-1)) \\
0, & \text{otherwise}
\end{cases}
$$

The functions \(W_{1+}\) and \(W_{1-}\) reflect immediate changes, capturing short-term temporal dynamics by comparing the current \(T_a\) value against its immediate predecessor. In contrast, \(W_{2+}\), \(W_{2-}\), \(W_{3+}\), and \(W_{3-}\) capture more extended temporal dependencies by considering the median or minimum of \(T_a\) over the last ten time steps.

---

These datasets provide a robust framework for evaluating and improving machine learning algorithms under dynamic and temporally dependent conditions.

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
