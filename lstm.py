import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
#Tensorflow and keras imports 
from tensorflow import keras 

#https://data.open-power-system-data.org/time_series/
#https://data.open-power-system-data.org/time_series/2020-10-06/README.md
# "Open Power System Data. 2020. Data Package Time series. Version
#     2020-10-06. https://doi.org/10.25832/time_series/2020-10-06. (Primary
#     data from various sources, for a complete list see URL)."
#Other Resources:
##https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/ 
#https://stackoverflow.com/questions/40684734/python-sklearn-preprocessing-minmaxscaler-1d-deprecation  

df = pd.read_csv("austria_power.csv")
#print(df.columns)
features = [
    "AT_load_actual_entsoe_transparency",
    "AT_load_forecast_entsoe_transparency",
    "AT_solar_generation_actual",
    "AT_wind_onshore_generation_actual"
]

#Explore. Nans? Weird Patterns?
outcome = "AT_price_day_ahead"
#print(df.isnull().sum())

df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
# plt.plot(df["utc_timestamp"], df[features[2]])
# plt.plot(df["utc_timestamp"], df[outcome])
# plt.show()

#2015 - 2017 dataset  
#Filter
df = df[(df["utc_timestamp"].dt.year >= 2015) & (df["utc_timestamp"].dt.year <= 2017)]
#print(df.describe())
#print(df.isnull().sum())

#Fill NaNs
df.fillna(method="bfill", inplace=True)
df.fillna(method="ffill", inplace=True)

#training_set
train_df = df[(df["utc_timestamp"].dt.year >= 2015) & (df["utc_timestamp"].dt.year <= 2016)]
test_df = df[(df["utc_timestamp"].dt.year >= 2017) & (df["utc_timestamp"].dt.year <= 2017)]



#Create feature and outcomes - train and test 
train_features = train_df[features]
train_outcome = train_df[outcome].values.reshape(-1, 1)
test_features = test_df[features]
test_outcome = test_df[outcome].values.reshape(-1, 1)
#print(train_features.describe())


# #Optional - Create offset 
# train_features = train_features.iloc[:-3]
# train_outcome = train_outcome[3:]
# test_features = test_features.iloc[:-3]
# test_outcome = test_outcome[3:]
# #Sanity check!
# print(len(train_features.index))
# print(len(train_outcome))
# print(len(test_features.index))
# print(len(test_outcome))

#Normalize the data features 
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)
train_outcome = scaler.fit_transform(train_outcome)
test_outcome = scaler.fit_transform(test_outcome)
#print(train_features)

#Todo - check generator 
generator = keras.preprocessing.sequence.TimeseriesGenerator(train_features, train_outcome, length=24, batch_size=32)

lstm_layer = keras.layers.LSTM(100, activation="relu", input_shape=(24, len(features)))
dense_layer = keras.layers.Dense(1, activation="relu")
#Model - can pass a list of layers 
model = keras.models.Sequential([lstm_layer, dense_layer])
model.compile(optimizer="adam", loss="mae")
print(model.summary())
model.fit_generator(generator, steps_per_epoch=1, epochs=50)

#Predict generator 
predict_generator = keras.preprocessing.sequence.TimeseriesGenerator(test_features, test_outcome, length=24, batch_size=32)
predictions = model.predict(predict_generator)
#print(predictions)
score = model.evaluate(predict_generator)
actual_test_outcomes = test_outcome[24:]
print(f"Model test score {score}")

start_index = 0
end_index = 1000
times = test_df["utc_timestamp"].iloc[24:]
plt.plot(times[start_index:end_index], predictions[start_index:end_index],  color="red", label="predictions")
plt.plot(times[start_index:end_index], actual_test_outcomes[start_index:end_index], color="blue", label="ground_truth")
plt.legend()
plt.show()


