# Kernel SVM 

# Importing the dataset
dataset = read.csv('output.csv')
dataset = dataset[2:184]

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$major, SplitRatio = 1)
training_set = subset(dataset, split == FALSE)
test_set = subset(dataset, split == TRUE)

# Feature Scaling is not required since all features (i.e. departments) carry equal weight

# Fitting Kernel SVM to the Training set
install.packages('e1071')
library(e1071)
classifier = svm(formula = major ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset_eval)

# Predicting the actual eval.csv results - final
# Reading the eval_data.csv file
dataset_eval = read.csv('eval_new.csv')
dataset_eval = dataset_eval[3:184]
# Predicting the final eval.csv results
y_pred = predict(classifier, newdata = dataset_eval)

