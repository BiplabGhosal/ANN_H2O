########## Loading Required Libraries

library(h2o)
library(caTools)

########## Data Loading
dataset <- read.csv('F:\\Work and Study\\Udemy Machine Learning\\Artificial_Neural_Networks\\Churn_Modelling.csv')
View(dataset)
str(dataset)
# Dataset is having the details of the bank customer

#1. RowNUmber #2. CustomerID #3. Surname #4. CreditScore #5. Geography #6. Gender #7. Age
#8. Tenure #9. Balance #10. NumberofProducts #11. HasCreditCard #12. IsActiveMember
#13. EstimatedSalary #14. Exited

# Objective of the model is to predict whether a customer is likely to exit the bank or not based on the data available

########### Data Cleaning & Preprocessing
#1. Removing RowNUmber, CusotmerID & Surname column as these columns donot have any impact on dependent variable
dataset <- dataset[,4:14]

#2. Converting Categorical Variables Geography and Gender into numeric
dataset$Gender <- factor(dataset$Gender,
                         levels = c('Female','Male'),
                         labels = c(1,2))
dataset$Gender <- as.numeric(dataset$Gender)

dataset$Geography <- factor(dataset$Geography,
                            levels = c('France','Spain','Germany'),
                            labels = c(1,2,3))
dataset$Geography <- as.numeric(dataset$Geography)

#3. Feature Scaling
dataset[-11] <- scale(dataset[-11])

#4. Train-Test Split
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set <- dataset[split,]
test_set <- dataset[!split,]

######### Building ANN using H2O
h2o.init(nthreads = -1) #Initiating H2o instance.Can connect to a server using ip/port, but here we are connecting to default instance.

classifier <- h2o.deeplearning(y='Exited',training_frame = as.h2o(training_set),
                               activation = 'Rectifier',hidden = c(6,6),
                               epochs = 100, train_samples_per_iteration = -2)

# Here using Rectifier as the Activation function, 2 hidden layers in the network and 6 neurons into each of hidden layers
# There is no thumb rule to take the number of neurons in the hidden layers, but we can take the number of neurons
# as average of number of dependent variables & independent variables. For our case we have 1 dependent variable and 10 independent variables
# hence average is rounded of to 6. We can tune these parameters using cross validation
# train_samples_per_iteration refers to the number of observations after which we want to update the weights.
# This value can be 1, if we want to update the weight after each iteration passing through ANN or it can be more than 1, meaning it will update weights after passing stipulated iterations.
# h2o package can auto-tune the train_samples_per_iteration by passing -2. (automatic parameter tuning)

######## Prediction on test set using ANN model
prob_pred <- h2o.predict(classifier, newdata = as.h2o(test_set[-11]))

#Choosing threshold of 0.5.If probability greater than 0.5, customer is likely to leave the bank
y_pred <- (prob_pred > 0.5)
y_pred <- as.vector(y_pred)

#Creating confusion matrix
cm <- table(test_set[,11],y_pred)

#Calculating Accuracy
accuracy <- (1531+208)/(2000) #86.9%

h2o.shutdown() #shutting down h2o instance
