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

# Fitting Random Forest Classification to the Training set
install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-183],
                          y = training_set$major,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-183])


