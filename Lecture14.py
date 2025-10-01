'''
Support Vector Machine:
* Tries to classify data into categories
* Tries to find a decision boundary that maximizes the margin between classes
* Tries to find a decision boundary that maximizes the margin between classes
* Most "cushion" for the new data

Linear Separability
*A space is linearly seperable if you can find this separating structure neatly

Kernal trick = when we want to project vectors into higher dimensional spaces, we use inner products
Kernal trick can get the transformation of inner products

SVMs are farily powerful and explainable models
They can sometimes even work on non-tabular data

Radial basis function:
* Common for SVMs
* Can tune with gamma parameter
* Very popular

PCA
* Takes linear combinations of input vectors to retain information but reduce dimensions
* Not super interpretable
* Some similarities to our SVM process but different purpose
* Just know linear combinations of inputs can generate some useful information for a machine learning model 

Dataset:
* Dates. Classification Task
Type of Machine Learning:
* Supervised
Data split:
*10%
Model evaluation:
*Accuracy

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dates.csv")
# print(df.columns)
# plt.scatter(df["date_length"], df["date_diameter"])
# plt.show()
# plt.clf()

date_classes = ["Ajwa","Medjool"]
df["class_code"] = 0
medjool_indexes = df.loc[df["class"] == "Medjool"].index.tolist()
df.loc[medjool_indexes, "class_code"] = 1
# print(df["class_code"])

outcome_df = df["class_code"]
feature_df = df.drop(["class","class_code","color"], axis=1)

scaler = MinMaxScaler()
feature_df = scaler.fit_transform(feature_df)

training_features, test_features, training_outcomes, test_outcomes = train_test_split(feature_df, outcome_df, test_size = 0.1)

svm = SVC() #Class
svm.fit(training_features, training_outcomes)
test_accuracy_score = svm.score(test_features, test_outcomes)
training_accuracy_score = svm.score(training_features, training_outcomes)

print(f"Training accuracy: {training_accuracy_score}")
print(f"Test accuracy: {test_accuracy_score}")

#Test features
print(test_features)
predictions = svm.predict(test_features)
print("Predictions")
print(predictions)
print("Decision Function")
decision_function = svm.decision_function(test_features) #distance of each sample from the decision boundary
print(decision_function)