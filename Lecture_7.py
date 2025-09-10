#Lecture 7
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
