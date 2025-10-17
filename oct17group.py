'''
What do I see in the picture
Name: Daniel 
Story: Fixing the heater unit in the dark crawlspace betlow the house at time of disappearance
Notes: he has a welding mask and headphones

Name: Susan Henderson
Occupancy: Family Cook
Alibi: Making casserole for the family during the dissapearance
Notes: Drinking wine

Name: Elijah
Occupation: Mechanic
Alibi: Was working on the BMW in the garage during the disappearances
NOteS: Nothing unusual

Evidence: broken glass was found, glass came from three distinct sources
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
test = pd.read_csv("glass_test_set.csv")
training = pd.read_csv("glass_training_set.csv")
print(test)

# %%
print(training)
# %%
#features = training['RI','Na','Mg',"Al","Si","K","Ca", "Ba", "Fe"]
drop_features = ["Type_of_glass"]
features = training.drop(drop_features, axis=1)
outcomes = training['Type_of_glass']

#%%
training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=0.1)

model = tree.DecisionTreeClassifier(max_depth=7)

model.fit(training_features, training_outcomes)
test_accuracy_score = model.score(test_features, test_outcomes)
training_accuracy_score = model.score(training_features, training_outcomes)
print(f"Training accuracy {training_accuracy_score}")
print(f"Test accuracy {test_accuracy_score}")

#%%
#Predict a value
outcome_prediction = model.predict(test)
#Print
print(outcome_prediction)

#1 building window
#2 building_windows_non_float
#5 containers
#6 tableware
