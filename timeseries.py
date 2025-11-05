#%%
import pandas as pd
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt

#Read in our data
#Sensor 1
sensor_1_df = pd.read_csv("sensor_1.csv")
sensor_1_df["timestamp"] = pd.to_datetime(sensor_1_df["timestamp"])
sensor_1_df["timestamp"] = sensor_1_df["timestamp"].dt.tz_localize('utc')
sensor_1_df.drop(columns=["Unnamed: 0"], inplace=True)
#print(sensor_1_df.head)
# %%
# Sensor 2
sensor_2_df = pd.read_csv("sensor_2.csv")
sensor_2_df.drop(columns=["Unnamed: 0"], inplace=True)
print(sensor_2_df.head())
# %%
#Sensor 3
sensor_3_df = pd.read_csv("sensor_3.csv")
sensor_3_df.drop(columns=["Unnamed: 0"], inplace=True)
print(sensor_3_df.head())
# %%
# Sensor 4
sensor_4_df = pd.read_csv("sensor_4.csv")
sensor_4_df["timestamp"] = pd.to_datetime(sensor_4_df["timestamp"])
sensor_4_df["timestamp"] = sensor_4_df["timestamp"].dt.tz_localize('utc')
print(sensor_4_df.head())
# %%
# Step 1 - Associating a time stamp with the data that currently has no time stamp.
# Generally we want the timestamps in the same format

# Sensor 2 - add a timestamp
start_sensor_2 = pd.Timestamp(2022, 1, 1, 1, 0, 0)
print(start_sensor_2)
idx = pd.date_range(start_sensor_2, periods=len(sensor_2_df.index), freq="5min")
#print(idx)
s2_readings = sensor_2_df.iloc[:,0].values.tolist()
df_dict = {"timestamp": idx, "reading": s2_readings}
s2_df = pd.DataFrame(df_dict)
s2_df["timestamp"] = s2_df["timestamp"].dt.tz_localize('utc')
#print(s2_df.head)
# %%

# Sensor 3 - Associate a timestamp here too
end_sensor_3 = pd.Timestamp(2022, 12, 31, 23, 8, 0)
idx = pd.date_range(end=end_sensor_3, periods = len(sensor_3_df.index), freq="5min")
s3_readings = sensor_3_df.iloc[:,0].values.tolist()
df_dict = {"timestamp": idx, "reading": s3_readings}
s3_df = pd.DataFrame(df_dict)
#print(s3_df.head)

# Align sensor 3's time zone
# Convert S3 timezone from PRC to UTC
s3_df["timestamp"] = s3_df["timestamp"].dt.tz_localize("PRC")
s3_df["timestamp"] = s3_df["timestamp"].dt.tz_convert("UTC")
print(s3_df.head())
# %%
# Check for missing data
plt.scatter(sensor_1_df["timestamp"], sensor_1_df["reading"], color="g")
plt.scatter(s2_df["timestamp"], s2_df["reading"], color="b")
plt.scatter(s3_df["timestamp"], s3_df["reading"], color="r")
plt.scatter(sensor_4_df["timestamp"], sensor_4_df["reading"], color="y")
# %%
# Lets use Jan-March
start_date = pd.Timestamp(2022,1,1,0,0,0).tz_localize("UTC")
end_date = pd.Timestamp(2022,3,31,23,0,0).tz_localize("UTC")

#Decide our sampling interval - hourly
#Min, max, Mean for higher resolution sensors
#Start with sensor 2
s2_min = s2_df.resample("h", on="timestamp", origin=start_date).min()
s2_min.reset_index(inplace=True, drop=True)
s2_max = s2_df.resample("h", on="timestamp", origin=start_date).max()
s2_max.reset_index(inplace=True, drop=True)
s2_mean = s2_df.resample("h", on="timestamp", origin=start_date).mean()
s2_mean.reset_index(inplace=True)

#Sensor 3
s3_min = s3_df.resample("h", on="timestamp", origin=start_date).min()
s3_min.reset_index(inplace=True, drop=True)
s3_max = s3_df.resample("h", on="timestamp", origin=start_date).max()
s3_max.reset_index(inplace=True, drop=True)
s3_mean = s3_df.resample("h", on="timestamp", origin=start_date).mean()
s3_mean.reset_index(inplace=True)

#Sensor 4
s4_df = sensor_4_df
s4_min = s4_df.resample("h", on="timestamp", origin=start_date).min()
s4_min.reset_index(inplace=True, drop=True)
s4_max = s4_df.resample("h", on="timestamp", origin=start_date).max()
s4_max.reset_index(inplace=True, drop=True)
s4_mean = s4_df.resample("h", on="timestamp", origin=start_date).mean()
s4_mean.reset_index(inplace=True)
#%%
#Create our merged df
new_df = s4_mean.copy()
new_df = new_df.rename(columns={"reading": "s4_mean"})
#print(new_df.head())
merge_options = [s2_min, s2_max, s2_mean, s3_min, s3_max, s3_mean, s4_min, s4_max]
merge_names = ["s2_min", "s2_max", "s2_mean", "s3_min", "s3_max", "s3_mean", "s4_min", "s4_max"]

for i in range(0, len(merge_options)):
    sub_df = merge_options[i]
    sub_df.fillna(method="bfill", inplace=True)
    sub_df.fillna(method="ffill", inplace=True)
    sub_df.dropna(inplace=True)
    new_df = pd.merge_asof(new_df, sub_df, direction="nearest")
    new_df = new_df.rename(columns={"reading": f"{merge_names[i]}"})

print(new_df.head())
# %%
