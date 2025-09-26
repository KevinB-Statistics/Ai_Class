'''


Due October 8th, 2025

You are going to explore different models we have discussed in class using the fuel_end_use.csv in the "Datasets" tab of class.

Instructions:

    Try to classify the end use of the energy by the GHG emissions and temperature​
    Pick 3 models​
    Try 3 different hyperparameter settings on each. At least one hyperparameter must be a property of the underlying model (like tree size, kernel type, etc) ​
    Get the overall training and test accuracy for each of the 9 models​ (PLEASE report in a table)
    Also get the per-class prediction accuracy for each model (per class accuracy or F1 score for each of the three outcome classes)​
    2-3 page write-up on what models performed best/worst, and conjecture as to why

Deliverables:

    Write-up​ (18 points)
    Code files (2 points) (But If missing, 0 on the entire project)

Write-Up Instructions:

Reporting section that states:​

    Each model you used​ (2 points)
    Every hyperparameter you changed on the model and what is in (very brief explanation of what it does/how it's used)​ (4 points)
    Overall training and test accuracy for each of the 9 overall models​ (2 points)
    Per-class prediction accuracy or F1 score for each of the 9 models (3 outcome classes per model)​ (2 points)

Conclusion section that states:​

    What model of the 9 was best​ (2 points)
    What model of the 9 was worst​ (2 points)
    Conjecture as to why certain models did well or poorly​ (2 points)
    What you learned from the project​ (2 points)
    About 2-3 pages in length.

Models you can use:

    Linear Regression (any kind that works, like Lasso, Polynomial, etc)​
    Bayes Classifier​
    Decision Trees​
    Random Forests​
    Any Boosting or Bagging (AdaBoost, Bagging, Gradient Boost, etc)​
    SVM (SVC or LinearSVC)

Hyperparameters you can change (including but not limited to):

    How much data you sample​
    Test/Train Split percentage​

Examples of Model-level hyperparameters – Can find in Sci-Kit learn ​

    Decision Tree: criterion (gini or entropy, etc), max_depth, max_features, max_leaf nodes, etc.​
    Random Forest: criterion, n_estimators, max_depth, max_features, random_state, etc)​
    AdaBoost: n_estimators, learning rate, algorithm ​
    SVMs: Type of Kernel, Kernel function parameters, etc​
    Many more

In class, we will be going over how to do a lot of the data wrangling for this dataset. Feel free to reference the lecture notes/recording from this class session to complete the assignment.

'''