'''
Assignment 3 - Kevin Bui
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn. metrics import confusion_matrix, f1_score

#Load in the dataset
df = pd.read_csv("fuel_end_use.csv")
# print(df.head())
print(df["END_USE"].unique())
outcomes = ["Process Heating", "CHP and/or Cogeneration Process", "Conventional Boiler Use"]
features = ["Coal","Diesel","Natural_gas","Other","Residual_fuel_oil", "Temp_degC", "Total"]

#Convert to numeric
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors = "coerce")

#Drop the bad values
df = df.dropna()

#Separate data by end use
ph_df = df.loc[df["END_USE"] == outcomes[0]]
chp_df = df.loc[df["END_USE"] == outcomes[1]]
boil_df = df.loc[df["END_USE"] == outcomes[2]]

#Box plot
def plot_var(variable):
    plt.boxplot([ph_df[variable], chp_df[variable], boil_df[variable]])
    plt.title(variable)
    plt.show()
    plt.clf()

def prep_dataset(df, features, outcome):
    feature_df = df[features]
    outcome_df = df[outcome]
    #Encode the label
    encoder = LabelEncoder()
    outcome_df = encoder.fit_transform(outcome_df)
    print("Classes")
    print(encoder.classes_)
    #Normalize the data
    scaler = MinMaxScaler()
    feature_df = scaler.fit_transform(feature_df)
    #Split into training and test setes
    train_in, test_in, train_out, test_out = train_test_split(feature_df, outcome_df, test_size=0.2, random_state=123)
    return train_in, test_in, train_out, test_out

train_in, test_in, train_out, test_out = prep_dataset(df, features,"END_USE")
print(train_in[0])

# For reporting class names aligned to encoded labels
encoder_for_reporting = LabelEncoder().fit(df["END_USE"])
label_order = np.unique(test_out)
class_names = encoder_for_reporting.inverse_transform(label_order)


def run_model(model, train_in, test_in, train_out, test_out):
    model.fit(train_in, train_out)
    test_accuracy_score = model.score(test_in, test_out)
    training_accuracy_score = model.score(train_in, train_out)
    print(f"Training accuracy {training_accuracy_score}")
    print(f"Test accuracy {test_accuracy_score}")
    print("Per Class Accuracy")
    test_predictions = model.predict(test_in)
    matrix = confusion_matrix(test_out, test_predictions, labels=label_order)
    class_accuracies = matrix.diagonal()/matrix.sum(axis=1) #diagnoal is correctly classified individuals (percentage of actually predicted individuals)
    print(class_accuracies)
    return training_accuracy_score, test_accuracy_score, class_accuracies, test_predictions


######## Decision tree, random forest, SVM #########
results = []

# Decision trees
dt_param_list = [
    {"criterion": "gini", "max_depth": None, "min_samples_leaf": 1, "random_state":123},
    {"criterion": "entropy", "max_depth": 5, "min_samples_leaf": 2, "random_state":123},
    {"criterion": "gini", "max_depth": 3, "min_samples_leaf": 1, "random_state":123}
]

for i, p in enumerate(dt_param_list, 1):
    model = DecisionTreeClassifier(**p)
    tr_acc, te_acc, per_class_acc, test_pred = run_model(model, train_in, test_in, train_out, test_out)
    f1s = f1_score(test_out, test_pred, average=None, labels=label_order)
    row = {"Model": f"DecisionTree_{i}",
           "Params": p,
           "Train_Accuracy": tr_acc,
           "Test_Accuracy": te_acc}
    for j, cls in enumerate(class_names):
        row[f"PerClassAcc[{cls}]"] = per_class_acc[j]
        row[f"F1[{cls}]"] = f1s[j]
    results.append(row)

#random forests

rf_param_list = [
    {"n_estimators":200, "max_depth": None,"max_features":"sqrt", "random_state":123},
    {"n_estimators":500, "max_depth": 5,"max_features":"sqrt", "random_state":123},
    {"n_estimators":300, "max_depth": None,"max_features":None, "random_state":123}
]
for i, p in enumerate(rf_param_list, 1):
    model = RandomForestClassifier(**p)
    tr_acc, te_acc, per_class_acc, test_pred = run_model(model, train_in, test_in, train_out, test_out)
    f1s = f1_score(test_out, test_pred, average=None, labels=label_order)
    row = {"Model": f"RandomForrest_{i}",
           "Params": p,
           "Train_Accuracy": tr_acc,
           "Test_Accuracy": te_acc}
    for j, cls in enumerate(class_names):
        row[f"PerClassAcc[{cls}]"] = per_class_acc[j]
        row[f"F1[{cls}]"] = f1s[j]
    results.append(row)

#SVM
svm_param_list = [
    {"kernel": "linear", "C":1.0},
    {"kernel": "rbf", "C":1.0, "gamma":"scale"},
    {"kernel": "poly", "C": 1.0, "degree":3, "coef0": 1},
]
for i, p in enumerate(svm_param_list, 1):
    model = SVC(**p)
    tr_acc, te_acc, per_class_acc, test_pred = run_model(model, train_in, test_in, train_out, test_out)
    f1s = f1_score(test_out, test_pred, average=None, labels=label_order)
    row = {"Model": f"SVM_{i}",
           "Params": p,
           "Train_Accuracy": tr_acc,
           "Test_Accuracy": te_acc}
    for j, cls in enumerate(class_names):
        row[f"PerClassAcc[{cls}]"] = per_class_acc[j]
        row[f"F1[{cls}]"] = f1s[j]
    results.append(row)

#Results table
results_df = pd.DataFrame(results).sort_values("Test_Accuracy", ascending = False).reset_index(drop=True)
print(results_df)
results_df.to_csv("Assignment_3_model_results.csv", index=False)