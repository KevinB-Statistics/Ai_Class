#Lecture 4 Friday August 29th & Lecture 5 Wednesday September 2nd
'''
We are going to be dealing with excel files
'''

#Explore.py
#%%
import pandas as pd

#pyplot is sublibrary
import matplotlib.pyplot as plt 
import seaborn as sns


# %%
wine_df = pd.read_csv("winequality-white.csv", sep=";")
#print(wine_df.head())
# %%
### Print the columns - method 1
#print("Columns Index")
#print(wine_df.columns)

# %%
### #Print the columns - method 2
#print("Columns as a list")
#print(list(wine_df.columns))
# %%
#### Data summary
#print("Describe the data")
#print(wine_df.describe())

# %%
#### Indexing the data
#quality_series = wine_df['quality']
#print(quality_series)

# %%
#### Get a list of values from a column
#quality_list = wine_df['quality'].values.tolist()
#print("Quality List")
#print(quality_list)
# %%
### Using .loc
# .loc include start and end values
# Get everything in the quality column
#print("Entire quality column")
#print(wine_df.loc[:,"quality"])

#print("First and up to 5th-index quality values")
#print(wine_df.loc[0:5,"quality"])

#print("Everything up to 5th indexed row quailty value")
#print(wine_df.loc[:5,"quality"])

#print("Everything after 4890, quality")
#print(wine_df.loc[4890:,"quality"])

#print("Everything up to 5th indexed row quailty and sulphates")
#print(wine_df.loc[0:5, ["quality", "sulphates"]])


# %%
### Using iloc (can't pass column names excludes end values)
# Index row and columns
#print("Row index 1-2 and column indexes 1-4")
#print(wine_df  .iloc[0:3, 1:5])

#print("4th row index, first item")
#print(wine_df.iloc[4,0])

#print("All columns, row index 4")
#print(wine_df.iloc[4,:])

# %%
### Selecting a row or rows based on condition
#print("All rows where the quality is above 5")
#print(wine_df.loc[wine_df["quality"] > 5])

#print("How mnay rows have a quality above 5")
#print(len(wine_df.loc[wine_df["quality"] > 5].index))

### Logical and, &
### Logical or, |
#print("All rows where quality above 5 and sulphates above 0.45")
#sub_df = wine_df.loc[(wine_df["quality"]>5) & (wine_df["sulphates"] > 0.45)]
#print(sub_df.head())
#print("How many rows is this?")
#print(len(sub_df.index))

# %%
#NaN means not a number
# Finding and handle missing numbers in data frame
#print(wine_df.isna())
#print(wine_df.isna().value_counts()) # no missing data

#boolean_df = wine_df.isna()
#print(boolean_df.head())

# insert missing data and handle it
#print(wine_df.loc[0, 'quality'])
#new_df = wine_df
#new_df.head()
#new_df.loc[0,'quality'] = None
#print(new_df.loc[0,'quality'])

#Drop the NaN value
#print("Number of rows before dropping NaN value")
#print(len(new_df.index))

#Create the 'cleaned' df
#cleaned_df = new_df.dropna()
#Reset the index (Two methods)
#cleaned_df = cleaned_df.reset_index()
#cleaned_df.reset_index(inplace=True)

#print(cleaned_df.head())
#print("Number of rows after dropping NaN")
#print(len(cleaned_df.index))
#print(cleaned_df.loc[0,'quality']) #index zero doesn't exist anymore


# %%
###Correlation Heatmap
print(wine_df.corr())
plot = sns.heatmap(wine_df.corr())
#plt.savefig("correlation_plot.png")
#plt.show()
plt.clf()
# %%
### Scatter plot
dependent_var = "quality"
independent_var = "density"
plt.scatter(wine_df[independent_var],wine_df[dependent_var])
plt.title(f"Correlation between {independent_var} and {dependent_var}")
plt.xlabel(independent_var)
plt.ylabel(dependent_var)
plt.show()
plt.clf()

# %%
