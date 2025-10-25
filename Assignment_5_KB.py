'''
    Due October 27, 2025
    Same dataset (faults)​
    Report Test/Validation/Training Accuracy​
    Try three different neural network architectures ​
        Different number of layers, or​
        Different number of nodes, or​
        Different activation functions ​
    Deliverables​
        Code Files​ (4 points - but 0 for whole assignment if missing)
        Table:(16 points)
            NN model architecture description (4 points)
            Training accuracy for each mode (4 points)
            Validation accuracy for each mode (4 points)l, and
            Test accuracy​ for each model (4 points)
    No formal write-up.


'''

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
#----------------------------
# Load in dataset
#----------------------------
# Read in our faults csv
df = pd.read_csv("faults.csv")
# Our faults
faults = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
"Other_Faults"]
#-------------------------------------------
#Create our dataset
#--------------------------------------------
# Features - everything but the faults
features = df.drop(faults, axis=1)
# Outcomes - just the faults
outcomes = df[faults]

# Print both columns and shape
print(features.columns)
print(outcomes.columns)
print(features.shape)
print(outcomes.shape)
#%%
#--------------------
# Data preprocessing
#-------------------
# Normalization
scaler = MinMaxScaler()

# Numpy array
features = scaler.fit_transform(features)
outcomes = scaler.fit_transform(outcomes)

# 80/20 split with random state
train_x, test_x, train_y, test_y = train_test_split(features, outcomes,
test_size=0.2, random_state=123)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# %%
#--------------------------
# Model Setup 3 Architectures
#--------------------------

# Model 1 different activation function (leaky relu)
m1_input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
m1_layer_1 = keras.layers.Dense(30, activation="leaky_relu")
m1_layer_2 = keras.layers.Dense(15, activation="leaky_relu")
m1_output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")

model1 = keras.models.Sequential([m1_input_layer, m1_layer_1, m1_layer_2, m1_output_layer])
print(model1.summary())

#%%
# Model 2 different activation function (leaky relu) and added another layer
m2_input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
m2_layer_1 = keras.layers.Dense(30, activation="leaky_relu")
m2_layer_2 = keras.layers.Dense(15, activation = "leaky_relu")
m2_layer_3 = keras.layers.Dense(5, activation = "leaky_relu")
m2_output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")

model2 = keras.models.Sequential([m2_input_layer, m2_layer_1, m2_layer_2, m2_layer_3, m2_output_layer])
print(model2.summary())


#%%
# Model 3 different activation function (leaky relu), another layer, and more nodes
m3_input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
m3_layer_1 = keras.layers.Dense(64, activation="leaky_relu")
m3_layer_2 = keras.layers.Dense(32, activation="leaky_relu")
m3_layer_3 = keras.layers.Dense(16, activation = "leaky_relu")
m3_output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")

model3 = keras.models.Sequential([m3_input_layer, m3_layer_1, m3_layer_2, m3_layer_3, m3_output_layer])

print(model3.summary())


#%%
models = [("Model 1", model1), ("Model 2", model2), ("Model 3", model3)]

#%%
#----------------
# Compile the model.
#-----------------

epochs = 25
batch_size = 32
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_function = keras.losses.CategoricalCrossentropy()
metric_list = [keras.metrics.CategoricalAccuracy()]

callback = [
    keras.callbacks.EarlyStopping(patience=2, 
                                  monitor="val_acc",
                                  mode="max",
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("my_models/checkpoint.keras",
                                    monitor="val_acc", 
                                    mode = "max", 
                                    save_best_only = True)
                                    ]

results = []
histories = {}

for name, m in models:
    #Compile the model
    m.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)

    #Fit the model
    h = m.fit(train_x, train_y, epochs = epochs, batch_size=batch_size, validation_split=0.1, callbacks=callback, verbose = 0)

    histories[name] = h.history

    # evlauate the model
    train_metrics = m.evaluate(train_x, train_y, verbose=0)
    test_metrics = m.evaluate(test_x, test_y,  verbose=0)

    #save results
    results.append({
        "Model": name,
        "Train_acc": train_metrics["acc"],
        "Best_val_acc": max(h.history["val_acc"]),
        "Test_acc": test_metrics["acc"],
        "Train_loss": train_metrics["loss"],
        "Best_val_loss": min(h.history["val_loss"]),
        "Test_loss": test_metrics["loss"]
    })

#%%
# -------------------------------------
# Train the model
# --------------------------------------
history = model.fit(x=train_x, 
                    y=train_y, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    callbacks=callback, 
                    validation_split=0.1)


train_acc_epochs = history.history['acc']
val_acc_epochs = history.history['val_acc']

train_metrics = model.evaluate(train_x, train_y, verbose=0)
test_metrics = model.evaluate(test_x,  test_y,  verbose=0)

print(f"Final Train accuracy: {train_metrics['acc']:.4f}")
print(f"Final Test accuracy: {test_metrics['acc']:.4f}")
print(f"Best Val accuracy: {max(val_acc_epochs):.4f}")
print(f"Last-epoch VAL accuracy: {val_acc_epochs[-1]:.4f}")

#%%
#-------------------------
# Run a prediction
#-------------------------
predictions = model.predict(test_x)
# Convert back from one-hot vectors due to scikitlearn documentation for report
pred_y = np.argmax(predictions,axis=1)
out_y = np.argmax(test_y, axis=1)

#Get confusion matrix and classification
matrix = confusion_matrix(out_y,pred_y)
report = classification_report(out_y,pred_y)
class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
print(faults)
print(class_accuracies)
print(report)
model.save("my_model.h5")
