import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

#Load in the dataset
df = pd.read_csv("faults.csv")
print(df.head())

# Create ONE column for all the defect outcomes
faults = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]
df['fault'] = 0
print(df.columns)

#Create a categorical variable for each fault
#Inside of the new 'fault' column
for i in range(0,len(faults)):
    true_fault_index = df.loc[df[faults[i]] == 1].index.tolist()
    df.loc[true_fault_index,"fault"] = i+1

#Create our dataset - inputs and outcomes
drop_features = ["fault"] + faults #drops the defect outcomes
features = df.drop(drop_features, axis=1)
outcomes = df["fault"]

training_features, test_features, training_outcomes, test_outcomes = train_test_split(features,outcomes,test_size=0.1,random_state = 123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"Mean Accuracy fo the model: {mean_accuracy}")

############### Question 1 ###################
'''
Pick only a subset of our input features, rather than all of them. Re-run the model (with a 10% test set). Note the accuracy. Did this increase or decrease? Why do you think this might be?
'''
subset_features = df[["Orientation_Index","LogOfAreas","Steel_Plate_Thickness"]]

training_features, test_features, training_outcomes, test_outcomes = train_test_split(subset_features,outcomes,test_size=0.1, random_state=123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"Mean Accuracy for the Subset 1 model: {mean_accuracy}")

############### Question 2 ###################
'''
Pick a different subset of features (it doesn’t have to be 100% different, but at least some features should be), and re-run the model with a 10% test set. Note the accuracy. Did this increase or decrease? Why do you think this might be?
'''
subset_features2 = df[['TypeOfSteel_A300', 'TypeOfSteel_A400']]
subset_features2.head()

training_features, test_features, training_outcomes, test_outcomes = train_test_split(subset_features2,outcomes,test_size=0.1, random_state=123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"Mean Accuracy for the Subset 2 model: {mean_accuracy}")

############### Question 3 ###################
'''
Pick one of the models from #1 or #2 and re-run it with a 5% test set and a 20% test set. Did your accuracy increase or decrease? Why?
'''
# 5% test set
training_features, test_features, training_outcomes, test_outcomes = train_test_split(subset_features2,outcomes,test_size=0.05, random_state=123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"5% Test Set - Mean Accuracy for the Subset 2 model: {mean_accuracy}")

# 20% test set
training_features, test_features, training_outcomes, test_outcomes = train_test_split(subset_features2,outcomes,test_size=0.2, random_state=123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"20% Test Set - Mean Accuracy for the Subset 2 model: {mean_accuracy}")

############### Question 4 ###################
'''
Pick your best overall model so far. Get the predictions for all of your test data on the model. Figure out which type of defect it had the overall best predictions on (like Pastry, Dirtiness, etc) and report it. Why do you think this was the case (and I don’t expect you to be a steel expert, really vague answers on this one are okay)
'''

subset_features = df[["Orientation_Index","LogOfAreas","Steel_Plate_Thickness"]]
print(subset_features.head())

training_features, test_features, training_outcomes, test_outcomes = train_test_split(subset_features,outcomes,test_size=0.1, random_state=123)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
test_predictions = bayes_classifier.predict(test_features)
report = classification_report(test_outcomes, test_predictions)
print(report)

#Classification report
fault_names = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]
labels = list(range(1,8))
test_predictions = bayes_classifier.predict(test_features)
report = classification_report(test_outcomes, test_predictions,labels=labels, target_names=fault_names,output_dict=True)

report_df = pd.DataFrame(report).T
print(report_df)
