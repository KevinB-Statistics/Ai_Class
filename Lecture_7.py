#Lecture 7 and 8
'''
Bayesian models:

Dataset:
- Steel faults. Input features = measurements. Output outcome = type of fault 

Type of Machine Learning
-Supervised

Data split
-Recommend Training/Test in this case

Model Evaluation
-Accuracy

Predict on all test set
Don't use defects as input (Z_scratch, stains, bumps)


'''
#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

#Load in the dataset
df = pd.read_csv("faults.csv")
print(df.head())
# %%

#Visualize the correlation
print(df.corr())
plot = sns.heatmap(df.corr())
plt.show()
#plt.savefig("plot.png")
plt.clf()
# %%
# Create ONE column for all the defect outcomes
faults = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]
df['fault'] = 0
print(df.columns)


# %%
#Create a categorical variable for each fault
#Inside of the new 'fault' column
for i in range(0,len(faults)):
    true_fault_index = df.loc[df[faults[i]] == 1].index.tolist()
    df.loc[true_fault_index,"fault"] = i+1
print(df['fault'])
# %%
#Create our dataset - inputs and outcomes
drop_features = ["fault"] + faults #drops the defect outcomes
features = df.drop(drop_features, axis=1)
outcomes = df["fault"]
print(features.head())
print(outcomes.head())
# %%

training_features, test_features, training_outcomes, test_outcomes = train_test_split(features,outcomes,test_size=0.1)
bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f"Mean Accuracy fo the model: {mean_accuracy}")

#Predict a random row
test_features .reset_index(inplace=True)
#Input to prediction model
number = 3
random_features = pd.DataFrame([test_features.iloc[number]])
random_features = random_features.drop(["index"],axis=1)
#Actual defect of the input
random_outcomes = test_outcomes.tolist()[number]
outcome_prediction = bayes_classifier.predict(random_features)
#Print it
#print("Input to the prediction model")
#print(random_features)
#print("Outcome")
#print(random_outcomes)
#print("Prediction")
#print(outcome_prediction[0])
#print(f"Predicted Fault: {faults[outcome_prediction[0]-1]}, Actual Fault: {faults[random_outcomes-1]}")

#More Scoring
test_predictions = bayes_classifier.predict(test_features.drop(["index"],axis=1))
report = classification_report(test_outcomes, test_predictions)
#print(report)

#Confusion matrix
matrix = confusion_matrix(test_outcomes, test_predictions)
print("Confusion Matrix")
print(matrix)

# %%
