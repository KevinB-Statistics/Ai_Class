#%%
#Imports and warning suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
#Tensorflow and keras imports
from tensorflow import keras
#import Input, layers, models, optimizers, losses, metrics
#Data Wrangling imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
#Graphing
import matplotlib.pyplot as plt

#Read in our faults csv
df = pd.read_csv("faults.csv")
#List out what our faults are
faults = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
"Other_Faults"]
#-------------------------------------------
#Create our dataset - features
#We don't have to make the new faults column this time
#Because they basically already have this one-hot encoded for us
#-------------------------------------------------
#Features - everything but the faults
features = df.drop(faults, axis=1)
#Outcomes - just the faults
outcomes = df[faults]
#Lets print both columns and shape
print(features.columns)
print(outcomes.columns)
print(features.shape)
print(outcomes.shape)
#%%
#--------------------
# Data preprocessing
#-------------------
#Lets also normalize everything
scaler = MinMaxScaler()
#Fit both - because it is expecting them in same format
#This function converts to numpy array
features = scaler.fit_transform(features)
outcomes = scaler.fit_transform(outcomes)
#And split into a training and test set
#Use a 20% training set
train_x, test_x, train_y, test_y = train_test_split(features, outcomes,
test_size=0.2)
# %%
#-----------------------
# Build the model
# Start with the layers we want to put
# inside of the model.
# Hyperparameters we are setting here = model structure
#-----------------------
#First - input layer. Provide the shape of the data (not batches)
input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
#Layers 1 and 2 - Dense layers
#Passing the number of nodes
#Also passing in the activation function (default-None)
#Can pass kernel and bias initializer
#Kernel -default glorot normal also called Xavier
#Sample from special normal distribution
#Lets make two layers
layer_1 = keras.layers.Dense(30, activation="relu")
layer_2 = keras.layers.Dense(15, activation="relu")
#Output layer - same but with softmax activation
#And same number of nodes as outputs
output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")
#Model - can pass a list of layers
model = keras.models.Sequential([input_layer, layer_1, layer_2, output_layer])
#Special function where we can print a summary of the model
# - very helpful to debug
print(model.summary())

#%%
#----------------
# Compile the model.
#-----------------
#Optional argument - an optimizer (we'll choose Adam)
#Loss function - pass it in
#Metrics - list of metrics you want to keep track of. Will
#Automatically keep track of the loss
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_function = keras.losses.CategoricalCrossentropy()
metric_list = [keras.metrics.CategoricalAccuracy()]
model.compile(optimizer=optimizer,
loss=loss_function, metrics=metric_list)

#%%
# -------------------------------------
# Train the model -- Biggest Time Sink
# --------------------------------------
#Expecting your train features and train outcomes
#Expecting epochs
#Expecting your batch size
#Expecting callbacks - later (and validation split later)
#Defaults to shuffling your data
callback_function = keras.callbacks.EarlyStopping(patience=2, monitor="val_loss",
restore_best_weights=True)
callback_function_2 = keras.callbacks.ModelCheckpoint(
"my_model/checkpoint_model",
monitor="val_loss",
mode="min",
save_best_only=True
)
epochs = 10
batch_size = 16
#history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs)
history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs,
callbacks=[callback_function, callback_function_2], validation_split=0.1)
print(history.history)
print("History available")
print(history.history.keys())
print("Loss")
print(history.history["loss"])
#print("Val Loss")
#print(history.history["val_loss"])
print("Validation Accuracy")
print(history.history["val_categorical_accuracy"])

#%%
#------------------
# Evaluate model
# x and y most important here
# Can also specify return dict
# Otherwise returns loss than metrics in order
#------------------
train_return_dict = model.evaluate(x=train_x, y=train_y, verbose=0)
print("Train Return: Loss, Accuracy")
print(train_return_dict)
test_return_dict = model.evaluate(x=test_x, y=test_y, verbose=0)
print("Test Return: Loss, Accuracy")
print(test_return_dict)
#-------------------------
# Run a prediction
#-------------------------
predictions = model.predict(test_x)
#print(predictions)
#print(test_y)
#Convert back from one-hot
pred_y = np.argmax(predictions, axis=1)
out_y = np.argmax(test_y, axis=1)
#Get the confusion matrix and classification
#Report from the sklearn functions
matrix = confusion_matrix(out_y, pred_y)
report = classification_report(out_y, pred_y)
class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
print(faults)
print(class_accuracies)
print(report)
loss_traces = ["loss", "val_loss"]
accuracy_traces = ["categorical_accuracy", "val_categorical_accuracy"]
#Graph training and validation loss and accuracy
for trace in accuracy_traces:
    plt.plot(history.history[trace])
    plt.legend(accuracy_traces)
    plt.show()
    #plt.xlabel
    #plt.y_label
    #plt.title
    plt.clf()