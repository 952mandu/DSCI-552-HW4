# DSCI 552 – Homework 4: Time Series Classification

Binary and multiclass classification of human activities from wireless sensor time series, using the UCI [AReM dataset](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29). Each of the 88 instances contains 6 time series recorded during one of 7 activities.

## Contents

- Time-domain feature extraction (min, max, mean, median, std, quartiles) with each series split into *l* segments
- Logistic regression with recursive feature elimination, tuned by stratified 5-fold cross-validation over (*l*, *p*)
- Confusion matrices, ROC/AUC, coefficient p-values, and case-control sampling
- L1-penalized logistic regression
- Multiclass classification: L1 multinomial regression, Gaussian and Multinomial Naive Bayes
- ISLR Exercises 4.8.3 and 4.8.7

## Results summary

| Task | Model | CV accuracy | Test result |
|---|---|---|---|
| Bending vs. other | L1 logistic regression (*l* = 1) | 1.000 | 0.947 accuracy |
| 7 activities | L1 multinomial regression (*l* = 1) | 0.884 | 0.053 error |
| 7 activities | Gaussian Naive Bayes (*l* = 2) | 0.857 | 0.053 error |
| 7 activities | Multinomial Naive Bayes (*l* = 5) | 0.825 | 0.263 error |

## How to run

1. Download and unzip the AReM data so the activity folders are at `AReM/AReM/` next to the notebook (the data is not included in this repo).
2. Install the dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn statsmodels`
3. Open `HW4.ipynb` and run all cells. The cross-validation loops take a few minutes.
