#Imports and warning suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#Tensorflow and keras imports
from tensorflow import keras
#Data Wrangling imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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
# print(features.columns)
# print(outcomes.columns)
# print(features.shape)
# print(outcomes.shape)

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

# print(train_x.shape)
# print(test_x.shape)
# print(train_y.shape)
# print(test_y.shape)

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

# Model 2 different activation function (leaky relu) and added another layer
m2_input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
m2_layer_1 = keras.layers.Dense(30, activation="leaky_relu")
m2_layer_2 = keras.layers.Dense(15, activation = "leaky_relu")
m2_layer_3 = keras.layers.Dense(5, activation = "leaky_relu")
m2_output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")

model2 = keras.models.Sequential([m2_input_layer, m2_layer_1, m2_layer_2, m2_layer_3, m2_output_layer])
print(model2.summary())

# Model 3 different activation function (leaky relu), another layer, and more nodes
m3_input_layer = keras.layers.Input(shape=(train_x.shape[-1], ))
m3_layer_1 = keras.layers.Dense(64, activation="leaky_relu")
m3_layer_2 = keras.layers.Dense(32, activation="leaky_relu")
m3_layer_3 = keras.layers.Dense(16, activation = "leaky_relu")
m3_output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")

model3 = keras.models.Sequential([m3_input_layer, m3_layer_1, m3_layer_2, m3_layer_3, m3_output_layer])

print(model3.summary())

models = [("Model 1", model1), ("Model 2", model2), ("Model 3", model3)]

#----------------
# Compile the model.
#-----------------

callback = [
    keras.callbacks.EarlyStopping(patience=10, 
                                  monitor="val_loss",
                                  mode="min",
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("my_models/checkpoint.keras",
                                    monitor="val_loss", 
                                    mode = "min", 
                                    save_best_only = True)
                                    ]

results = []
histories = {}

#Loop through the models

for name, m in models:
    epochs = 50
    batch_size = 16
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_function = keras.losses.CategoricalCrossentropy()
    metric_list = [keras.metrics.CategoricalAccuracy()]
    #Compile the model
    m.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list, run_eagerly=True)

    #Fit the model
    h = m.fit(train_x, train_y, epochs = epochs, batch_size=batch_size, validation_split=0.1, callbacks=callback, verbose = 1)

    histories[name] = h.history

    # evlauate the model
    train_metrics = m.evaluate(train_x, train_y, verbose=0, return_dict=True)
    test_metrics = m.evaluate(test_x, test_y, verbose=0, return_dict=True)

    # results
    results.append({
        "Model": name,
        "Train_accuracy": train_metrics["categorical_accuracy"],
        "Best_validation_accuracy": max(h.history["val_categorical_accuracy"]),
        "Test_acc": test_metrics["categorical_accuracy"],
        "Train_loss": train_metrics["loss"],
        "Best_val_loss": min(h.history["val_loss"]),
        "Test_loss": test_metrics["loss"]
    })
    # save model
    m.save(f"my_models/{name.replace(' ','_')}.keras")

# Printing csv
df = pd.DataFrame(results)

descriptions = {
    "Model 1": "Base model: 2 hidden layers (30,15) using Leaky ReLU",
    "Model 2": "Added 3rd hidden layer (5 nodes) with Leaky ReLU",
    "Model 3": "Increased number of nodes for each layer (64,32,16) with Leaky ReLU"
}

df["Architecture Description"] = df["Model"].map(descriptions)

df = df.rename(columns={
    "Model": "NN Model Architecture",
    "Train_accuracy":"Training Accuracy",
    "Best_validation_accuracy": "Best Validation Accuracy",
    "Test_acc":"Test Accuracy"
})

cols = ["NN Model Architecture",
        "Architecture Description",
        "Training Accuracy", "Best Validation Accuracy", "Test Accuracy"]

df = df[cols]
print("---Neural Network Performance Summary---")
print(df.to_markdown(index=False, floatfmt=".4f"))

df.to_csv("NN_results_table.csv", index=False)
