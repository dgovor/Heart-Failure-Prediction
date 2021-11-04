# Heart-Failure-Prediction
Machine Learning model for heart failure prediction using LGBM Classifier.

## Description

This code was written in Python using a dataset provided on [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction). The dataset combines many observations provided by Cleveland, Hungary, Switzerland, VA Long Beach, and Stalog (Heart) Data Set.
The proposed ML model was developed in order to represent one of the possible solutions for the problem.

## Dataset

The provided dataset consists of 918 samples with 11 features each. The label represents whether a patient has a heart disease. This dataset is split using train_test_split into two subsets of data. One subset contains 80% of the original data and is used to train our model, second subset that is called validation subset contains the remaining 20% and is used to validate our model. The validation dataset is used to provide us with an understanding of the efficiency of our design.

## Data Preprocessing

Data preprocessing consists of the following steps:
* Check if the dataset bears any missing values;
* All categorical features are encoded;
* Check the correlation between features;
* Normalize the data.

## Results

To obtain the results it was decided to use LGBM Classifier. This method provided good results with accuracy of approximately 91%. Which suggests that the model works properly obtaining correct predictions for the vast majority of the data samples.

<p align="center"> <img src="Confusion matrix.png" />
