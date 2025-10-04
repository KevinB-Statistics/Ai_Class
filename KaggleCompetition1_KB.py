'''
Kaggle competition
'''

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
medical_train = pd.read_csv("ML_Class/train.csv")
medical_test = pd.read_csv("ML_Class/test.csv")

medical_train_encoded = pd.get_dummies(medical_train, columns=['sex', 'smoker', 'region'], drop_first=True)
medical_test_encoded = pd.get_dummies(medical_test, columns=['sex', 'smoker', 'region'], drop_first=True)

y_train = medical_train_encoded["charges"]
x_train = medical_train_encoded.drop(columns=["charges"])

#test features
x_test = medical_test_encoded.drop("ID",axis=1)
x_test = x_test.reindex(columns=x_test.columns, fill_value = 0)


#%%

training_features, test_features, training_outcomes, test_outcomes = train_test_split(x_train,y,test_size=0.1)

#Repeat the model training process, with training/test set
linear_regression_model = LinearRegression()
linear_regression_model.fit(training_features, training_outcomes)
y_hat = linear_regression_model.predict(test_features)
print("R^2:", r2_score(test_outcomes, y_hat))
print("MAE:", mean_absolute_error(test_outcomes, y_hat))

y_pred = linear_regression_model.predict(x_test)
# %%
submission = pd.DataFrame({
    'ID': medical_test_encoded['ID'],
    'charges': y_pred.astype(float)})
submission.to_csv('submission.csv', index=False)
print("Submission file successfully created!")
# %%
