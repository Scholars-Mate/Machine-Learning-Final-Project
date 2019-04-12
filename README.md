# CS4720: Introduction to Machine Learning and Pattern Recognition Final Project

This project will investigate the usage of Principle Component Analysis (PCA)
and Linear Discriminant Analysis (LDA) with different data sets, and compare
their performance (accuracy) with different classification methods (such as
K-nearest neighbor or logistic regression).

## Data Sets

There are three data sets to be used for this project. These were pulled from the
[UCI Machine Learning Repository] and are in the `datasets/` directory of this
project.

* [Dorothea Data Set]
* [Gisette Data Set]
* [Dexter Data Set]

These data sets were chosen for their extremely high dimensionality ranging from
5,000 features to 100,000 features. They also include "probe" features with no
predective power. Ideally, PCA and LDA will be able to remove these features.
The data sets also included test data with no provided labels. These data sets
are unused and removed for this project.

[UCI Machine Learning Repository]: http://archive.ics.uci.edu/ml/index.php
[Dorothea Data Set]: http://archive.ics.uci.edu/ml/datasets/Dorothea
[Gisette Data Set]: http://archive.ics.uci.edu/ml/datasets/Gisette
[Dexter Data Set]: http://archive.ics.uci.edu/ml/datasets/Dexter
