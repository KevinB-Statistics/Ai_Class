'''
Kaggle competition
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
medical_train = pd.read_csv("ML_Class/train.csv")
medical_test = pd.read_csv("ML_Class/test.csv")

medical_train_encoded = pd.get_dummies(medical_train, columns=['sex', 'smoker', 'region'], drop_first=True)
medical_test_encoded = pd.get_dummies(medical_test, columns=['sex', 'smoker', 'region'], drop_first=True)

y = ["charges"]
x_train = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
x_test = medical_test_encoded.drop("ID",axis=1)


#Create our dataset - inputs and outcomes
train_features = medical_train_encoded.drop(y, axis=1)
train_outcomes = medical_train_encoded[y]
print(train_features.head())
print(train_outcomes.head())


#%%

training_features, test_features, training_outcomes, test_outcomes = train_test_split(train_features,train_outcomes,test_size=0.1)

#Repeat the model training process, with training/test set
linear_regression_model = LinearRegression()
linear_regression_model.fit(train_features, train_outcomes)
y_pred = linear_regression_model.predict(x_test)

# %%
submission = pd.DataFrame({'ID': pd.Series(medical_test_encoded['ID']), 'charges':pd.Series(y_pred)})
bro = pd.Series(medical_test_encoded['ID'])
print(bro)
# submission.to_csv('submission.csv', index=False)
# print("Submission file successfully created!")
# %%
