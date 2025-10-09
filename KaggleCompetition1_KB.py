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

#import data
medical_train = pd.read_csv("ML_Class/train.csv")
medical_test = pd.read_csv("ML_Class/test.csv")

# encoding
medical_train_encoded = pd.get_dummies(medical_train, columns=['sex', 'smoker', 'region'], drop_first=True)
medical_test_encoded = pd.get_dummies(medical_test, columns=['sex', 'smoker', 'region'], drop_first=True)

#targets and features
y_train = medical_train_encoded["charges"]
x_train = medical_train_encoded.drop(columns=["charges"])

#aligning test to training columns
x_test = medical_test_encoded.drop("ID",axis=1)
x_test = x_test.reindex(columns=x_test.columns, fill_value = 0)

training_features, test_features, training_outcomes, test_outcomes = train_test_split(x_train, y_train, test_size = 0.1)

############### OLS Model #################
linear_regression_model = Pipeline([
    ("scale", StandardScaler()),
    ("ols", LinearRegression())
])

linear_regression_model.fit(training_features, training_outcomes)
y_hat = linear_regression_model.predict(test_features)
print("R^2:", r2_score(test_outcomes, y_hat))
print("MAE:", mean_absolute_error(test_outcomes, y_hat))

# predict on test data
y_pred = linear_regression_model.predict(x_test)

################ Polnomial with ElasticNet and Log-transform Model #########################

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("enet", ElasticNetCV(
        l1_ratio=[0.1,0.25,0.3,0.5,0.7,0.75,0.9,0.95,1.0],
        cv=10,
        n_jobs=-1,
        max_iter=10000,
        tol=1e-4
    ))
])

linear_regression_model2=TransformedTargetRegressor(
    regressor=model,
    func=np.log1p, # y -> log(1+y)
    inverse_func=np.expm1 #backtransform
)

#fit
linear_regression_model2.fit(x_train,y_train)

#validation
y_hat2 = linear_regression_model2.predict(test_features)
print("R^2:", r2_score(test_outcomes, y_hat2))
print("MAE:", mean_absolute_error(test_outcomes, y_hat2))

y_pred2 = linear_regression_model2.predict(x_test)

submission = pd.DataFrame({
    'ID': medical_test_encoded['ID'],
    'charges': y_pred2.astype(float)})
submission.to_csv('submission2.csv', index=False)
print("Submission file successfully created!")
