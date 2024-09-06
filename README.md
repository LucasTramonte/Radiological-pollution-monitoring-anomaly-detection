# Radiological-pollution-monitoring-anomaly-detection
This project focuses on continuous monitoring of radiological pollution using gamma photon counting sensors. The main objective is to apply unsupervised learning methods to analyze unlabeled datasets. The goal is to detect and flag anomalies as new samples arrive in the time series data.


## Contents
- [Requirements](#Requirements)     
- [Analysis](#Analysis)
- [References](#References)

## Requirements

```bash
python pip install -r requirements.txt
```
## Analysis

This project includes several datasets related to gamma radiation, hygrometry, atmospheric pressure, temperature, and synthetic noise. Below is a explanation of each dataset:

1. 2015_months_DebitDoseA.csv
Contains primary data on gamma radiation levels, with measurements for each month of 2015.
2. 2015_months_HYGR.csv
Hygrometry (humidity) data for 2015, organized by month.
3. 2015_months_PATM.csv
Atmospheric pressure data for 2015, organized by month.
4. 2015_months_TEMP.csv
Temperature data for 2015, organized by month.
5. shortMm_0909_1.csv
Synthetic Gaussian white noise data.

Note: The hygrometry, pressure, and temperature datasets are not used in the main algorithm but demonstrate other potential applications of the method.

## References

- [1] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding,” in Proc. of the 24th ACM SIGKDD, 2018. 
- [2] L. Poirier-Herbeck, E. Lahalle, N. Saurel and S. Marcos, “Unknown length motif discovery methods in environmental monitoring  time series,” in Proc. of ICECET 2022. 
- [3] M. Lavielle, “Using penalized contrasts for the change-point problem,” Signal Processing, Vol.85, August 2005, pp. 1501-1510. 
- [4] K. P. Sinaga and M-S. Yang, “ Unsupervised K-means clustering algorithm ”, IEEEaccess vol. 8, 2020, p. 80716-80727. 