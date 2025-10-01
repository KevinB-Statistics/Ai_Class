#Gradient Descent
'''
Stochastic gradient descent - updates the model parameters after calculating the error for each individual training sample; this is sensitive to errors in the data and computationally expensive
Batch gradient descent - runs several different training examples in a batch and then averages the error across the batches. It then updates the parameters by taking the derivative of the average error with respect to the parameters (happy medium)

Algorithm behind most of the neural networks
incredibly popular in supervised learning

*Gradient descent is often sensitive to hyperparameters
-Model parameters are what we adjust in the gradient descent training algorithm
-But hyperparameters are what we, the programmer designer, set and don't change
-The error function we chose to use (MSE,MAE,etc) is a hyperparameter
-The step size we adjust by (in neural networks, called the learning rate) is a hyperparameter
-how many samples to use to compute error is hyper parameter
- when to stop is a hyper parameter

'''