#Help from the pandas documentation:
#https://pandas.pydata.org/docs/user_guide/

#The matplotlib documentation:
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

#And from here:
#https://stackoverflow.com/questions/39409866/correlation-heatmap

#%%
#Import the pandas dataset 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Read the dataset into the "wine_df" variable
wine_df = pd.read_csv("winequality-white.csv", sep=";")

#%%
#Get an overview of the data 
print(wine_df.head())

#%%
#Print the columns - method 1
print("Columns Index")
print(wine_df.columns)

#Print the columns - method 2
print("Columns as a List")
print(list(wine_df.columns))

#Describe the data 
print("Describe the data")
print(wine_df.describe())


#Indexing 
#Return a series with the 'quality' column of of the dataset
quality_series = wine_df['quality'] 
print("Quality series")
print(quality_series)


#Get a list of values from a column
quality_list = wine_df['quality'].values.tolist()
print("Quality List")
print(quality_list)


#loc 
#Get everything in quality 
print("Entire quality column")
print(wine_df.loc[:,"quality"])

#Get the first value in quality
print("First quality value")
print(wine_df.loc[0,"quality"])

print("First and up to 5th-indexed quality values")
print(wine_df.loc[0:5,"quality"])

print("Everything before and up to the 5th-indexed quality values")
print(wine_df.loc[:5,"quality"])

print("Everything after the 4890th-indexed quality value")
print(wine_df.loc[4890:,"quality"])

print("First through 5th-indexed quality and sulphates values")
print(wine_df.loc[0:5,["quality", "sulphates"]])

#Use iloc 
print("Row indexes 1-2 and column indexes 1-4")
print(wine_df.iloc[1:3, 1:5])

print("4th row index, first item")
print(wine_df.iloc[4, 0])

print("All columns, row index 4")
print(wine_df.iloc[4, :])



#Selecting a row based on a condition 
print("All rows where quality is above 5")
print(wine_df.loc[wine_df["quality"] > 5])
#How many rows is this?
print("How many rows is this?")
print(len(wine_df.loc[wine_df["quality"] > 5].index))

print("All rows where quality is above 5 and sulphates are above 0.45")
sub_df = wine_df.loc[(wine_df["quality"] > 5) & (wine_df["sulphates"] < 0.45)]
print(sub_df.head())
print("How many rows is this?")
print(len(sub_df.index))


#Find and Handle NaNs 
#We don't have an NaN values
print(wine_df.isna())
print(wine_df.isna().value_counts())


#Insert an NaN value
print(wine_df.loc[0, 'quality'])
new_df = wine_df
new_df.loc[0, 'quality'] = None
print(new_df.loc[0, 'quality'])

#Drop the na value 
print("Number of rows before dropping NaN")
print(len(new_df.index))
cleaned_df = new_df.dropna()
#Reset the index - in place this time 
cleaned_df.reset_index(inplace=True)
print("Number of rows after dropping NaN")
print(len(cleaned_df.index))
print(cleaned_df.loc[0, "quality"])



#Visualize 
#Correlation
print(wine_df.corr())
plot = sns.heatmap(wine_df.corr()) 
plt.show()
plt.clf()

#Scatter
plt.scatter(wine_df["quality"], wine_df["alcohol"])
plt.show()
plt.clf()


# %%
