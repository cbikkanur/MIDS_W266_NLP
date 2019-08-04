# Debiasing Personal Identities in Toxicity Classification

As Machine Learning models continue to be relied upon for making automated decisions, the issue of model bias becomes more and more prevalent. In this project, we approach training a text classification model and optimize on bias minimization by measuring not only the models performance on our dataset as a whole, but also how it performs across different subgroups. This requires measuring performance independently for different demographic subgroups and measuring bias bycomparing them to results from the rest of our data. We show how unintended bias can be detected using these metrics and how removing bias from a dataset completely can result in worse results.

### Data:
For this project, we used data from [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) competition from Kaggle platform.
[train_data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/download/train.csv): 
