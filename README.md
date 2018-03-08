# Classification_Implementations

The goal of this project was to implement the Naive Bayes classifier, k-NN algorithm, and Logistic Regression from scratch. We will test the models on the spambase dataset, predicting spam vs not spam.

## Data
The data consists of examples of spam and non-spam emails, of which there are 4508 training examples and 93 testing examples. The feature vector x is a 57-dimensional vector extracted from the email and y = 1 indicates a spam email. The data has been preprocessed such that the first 54-dimensions of each observation is binary and the last three dimensions are positive numbers. Ideally, we would use cross-validation on multiple partitions, but I am keeping it simple here with one training and one testing as the main goal is to implement the classifiers.