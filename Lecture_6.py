#Lecture 6 Monday, Sept 8th
'''
Linear regression
'''
#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Load in the dataset
wine_df = pd.read_csv("winequality-white.csv", sep=';')

#%%
plt.scatter(wine_df["alcohol"], wine_df["quality"])
plt.xlabel("alcohol")
plt.ylabel("quality")
plt.title("Alcohol vs Quality")
plt.show()


# %%
# Manual regression between alcohol and quality
# Get the variables we need to make calculations
x = "alcohol"
y = "quality"
n = len(wine_df.index)
sum_x = wine_df[x].sum()
sum_y = wine_df[y].sum()
x_mean = sum_x/n
y_mean = sum_y/n 
sum_x_times_y = (wine_df[x]*wine_df[y]).sum()
sum_x_squareds = wine_df[x].pow(2).sum()
print(f"n: {n}")
print(f"Sum of x values: {sum_x}")
print(f"Sum of y values: {sum_y}")
print(f"Mean of x value: {x_mean}")
print(f"Mean of y value: {y_mean}")
print(f"Sum of x times y: {sum_x_times_y}")
print(f"Sum of x squareds: {sum_x_squareds}")
# %%
Sxy = sum_x_times_y - (sum_x*sum_y)/n
Sxx = sum_x_squareds - (sum_x*sum_y)/n
print(f"Sxy: {Sxy} Sxx:{Sxx}")

# %%
#Get our line variables
m = Sxy/Sxx
b = y_mean - m*x_mean
print(f"Regression equation: y = {m}x + {b}")
# %%
#Lets predict a value
row = 1000
alcohol_val = wine_df.loc[row,"alcohol"]
quality_val = wine_df.loc[row,"quality"]
predicted_quality = alcohol_val*m + b
print(f"For alcohol value {alcohol_val}, the predicted quality was {predicted_quality} and actual quality is {quality_val}")
# %%
#Coefficient of determination
wine_df["predictions"] = wine_df["alcohol"]*m + b
residual_squares = (wine_df["quality"]- wine_df["predictions"]).pow(2).sum()
total_squares = (wine_df["quality"]-y_mean).pow(2).sum()
r_squared = 1 - residual_squares / total_squares
print(f"Coefficient of determination: {r_squared}")
# %%

#Doing linear regression with sklearn
linear_regression_model = LinearRegression()
#Expects a list of x variables because it can handle multiple regression
linear_regression_model.fit(wine_df[["alcohol"]], wine_df[["quality"]])
model_m = linear_regression_model.coef_[0][0]
model_b = linear_regression_model.intercept_[0]
print(f"For the Scikit learn regression model")
print(f"Regression equation: y={model_m}x + {model_b}")

#Predict the same value
predicted_model_quality = linear_regression_model.predict([[alcohol_val]])
print(f"The alcohol value {alcohol_val}, the predicted quality was {predicted_model_quality} and the actual quality is {quality_val}")

#Coefficient of determination
score = linear_regression_model.score(wine_df[["alcohol"]],wine_df[["quality"]])
print(f"Coefficient of determination: {score}")

#Plot the best fit line
plt.scatter(wine_df["alcohol"], wine_df["quality"])
plt.plot(wine_df["alcohol"], wine_df["predictions"], linestyle="solid")
plt.xlabel("alcohol")
plt.ylabel("quality")
plt.title("Alcohol vs Quality with best fit line")
plt.show()
# %%
