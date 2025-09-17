'''
September 15th, 2025
Classification Metrics
What belongs to what class?
How good is the model at determining this
Many considerations in metric calculation

Null hypothesis vs instance of class
A lot of these metrics depend on the class you "care" about

Model: Alarm clock model

Null hypothesis: it is not time to go to work yet (no alarm)
Positives: timestamps with alarm
Negatives: timestamps without alarm

Model metrics:
Accuracy - number of correctly predicted instances / total number of instances
Alarm clock model accuracy: number of times your alarm went off at the right time/total number of times you looked at

True positive: Correct prediction of the class you care about as the class you care about: number of predictions that the input is the class you care about that were the class you care about
Ex: Alarm suppose to go off at 5am your alarm correctly goes off at 5am.

True negative: Correct prediction of the not the class you care about as not the class you care about
Ex: Your alarm is not suppose to go off at 4am, your alarm is silent at 4am.

False positive: Incorrect prediction of the not the class you care about being the class you care about
Ex: Your alarm is not suppose to go off at 4am. Your alarm goes off at 4am.
Type 1 error

False negative: incorrect prediction of the class you care about as being not the class you care about
Ex: Alarm is suppose to go off at 5am. Your alarm doesn't go off at 5am.
Type 2 error

Which is worse? Depends

Metrics:
Confusion matrix
Per class accuracy: # of correctly predicted labels in class / total number of labels in class
Precision: Tp/(TP+FP) What percent of predicted positives were actually postiives
Recall TP/(TP+FN): What % of positive classes predicted accurately
F1 Score: 2*(precision*recall)/(precision + recall) (harmonic mean): generally, how good you are at actually predicting the class
OFten used as the metric in classification - rule of thumb > 0.8

False positive rate(Type 1 error): FP/(FP+TN): What % negative classes wrongly lassified
False negative rate(Type 2 error): FN/(FN+TP)
Specificity: TN/(TN+FP)

For assignments we are looking at individual classes within a classification problem, likely looking at per-class accuracy or F1 score
'''
##############################################################
'''
Information theory - quantifying uncertainty - looking at how 'surprising' or 'unsurprising' an event is
Tomorrow, the sun will rise (low information)
Tomorrow, it will be cloudy (Medium information)
Tomororw, you will be attacked by wolverines (High information ; very surprising)

Calculating the information of an event:
Information of event x = -log(p(x)) using log base 2
Inverse relationship between information of an event and probability
100% probability => 0 information

Information often written as h(x); information always positive

Entropy - How much information is in a random variable - basically looks at the variance of the random variable
Random variable x
K states in x
N instances of random variable x
Entropy of x = -sum{k}{n} p(k) * log(p(k))
Entropy gives you basically a measure of the average information in your distribution of the variable
Low entropy for average unsurprising distributions
High entropy for average suprising distributions

Information gain = entropy removed 

'''
###############################################################
'''
Decision Tree
Graphviz package installed on python
Basically flow chart

How do we create the trees?
We look for the best splits of the data to have efficient, useful trees

Improvements on decision tree
-pruning: uses your validation set to remove branches of the tree without killing accuracy
-random forest (ensemble model): train a bunch of decision trees often on a subset of the data, average the answer together or use some weighting/voting system
'''
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.ensemble import RandomForestClassifier

faults = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]

def read_in_dataset(faults):
    df = pd.read_csv("faults.csv")
    df["fault"] = 0
    #Create the categorical variable for each fault
    for i in range(0, len(faults)):
        #Indexes of faults
        true_fault_indexes = df.loc[df[faults[i]] == 1].index.tolist()
        df.loc[true_fault_indexes, "fault"] = i+1
    return df

df = read_in_dataset(faults)
print(df.head())
    
#Create the training and test set
drop_features = ["fault"] + faults
features = df.drop(drop_features, axis=1)
outcomes = df["fault"]
training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=0.1)
model = tree.DecisionTreeClassifier(max_depth=5)
#model = RandomForestClassifier()
model.fit(training_features, training_outcomes)
test_accuracy_score = model.score(test_features, test_outcomes)
training_accuracy_score = model.score(training_features, training_outcomes)
print(f"Training accuracy {training_accuracy_score}")
print(f"Test accuracy {test_accuracy_score}")
# #Visualize the tree
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("steel_tree")

#Predict a random value
test_features.reset_index(inplace=True)
# test_outcomes.reset_index(inplace=True)
number = 3
random_features = pd.DataFrame([test_features.iloc[number]])
random_features = random_features.drop(["index"], axis=1)
random_outcome = test_outcomes.tolist()[number]
outcome_prediction = model.predict(random_features)
#Print
print("Features")
print(random_features)
print(random_outcome)
print(outcome_prediction[0])
print(f"Predicted Fault: {faults[outcome_prediction[0]-1]}, Actual Fault {faults[random_outcome-1]}")