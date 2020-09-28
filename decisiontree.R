# Random Forest Classification

# Importing the dataset
dataset = read.csv('training.csv')
dataset = dataset[2:184]

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$major, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling is not required since all features (i.e. departments) carry equal weight


# Fitting Decision Tree Classification to the Training set
install.packages('rpart')
library(rpart)
classifier = rpart(formula = major ~ .,
                   data = training_set)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-183])


