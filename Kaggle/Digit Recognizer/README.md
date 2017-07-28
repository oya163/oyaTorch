# Digit Recognizer using PyTorch

This program recognizes hand-written digits using
LeNet with custom class which reads csv dataset.
I wrote this program to participate in Digit Recognizer Kaggle Competition. The accuracy is over 98%.

In the training set, the data is divided into labels and images.
In the testing set, the data is composed of only images.

This program reads training/testing dataset, trains the neural network, evaluates the data in testing set and creates a csv file, which is ready to be submitted to Kaggle.

PS. You have to download the dataset (train.csv/test.csv) from 
https://www.kaggle.com/c/digit-recognizer/data and save into folder named "input-files" in order to run this program smoothly.
