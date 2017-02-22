#Data PreProcessing

#Import the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

# Splitting the dataset into the training set and Test set
# instal.package('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# train_set[, 2:3] = scale(train_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
