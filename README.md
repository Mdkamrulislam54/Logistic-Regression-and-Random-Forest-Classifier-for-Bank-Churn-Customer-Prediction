# Logistic-Regression-and-Random-Forest-Classifier-for-Bank-Churn-Customer-Prediction


Bank_Churn_Customer_Project (1).ipynb
Bank_Churn_Customer_Project (1).ipynb_
Bank Churn Customer Analysis: Data Preparation & EDA (Exploratory Data Analysis)
The Situation
You've been hired as a Data Scientist/Data Analyst for a Bank.

The product team at the bank has noticed an uptick in Customer Churn and a decline in growth, and they want to find ways to reduce churn and appeal to new customers.

You've been asked to prepare and explore a set of customer data that will be used for two Machine Learning Projects: Churn Prediction & Customer Segmentation.

|| Cleaning and Exploring Bank Churn Customer Data to Prepare it for Machine Learning Models including Classification (Supervised) & Clustering (Unsupervised) ||

Objective 1: Importing & QA the Data
Our first objective is to Import & Join Two Customer Data Tables, then Remove Duplicate Rows & Columns and Fill in Missing Values.


[ ]
# Importing the Required Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

[ ]
churn_cust_info = pd.read_excel("Bank_Churn_Messy.xlsx")
churn_cust_info.head()


[ ]
# First, make sure your Google Drive is mounted in Colab:
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive

[ ]
# churn_cust_info = pd.read_excel("Bank_Churn_Messy.xlsx")
churn_cust_info = pd.read_excel("/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/Bank_Churn_Messy.xlsx")

churn_cust_info.head()


[ ]
churn_cust_info.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10001 entries, 0 to 10000
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10001 non-null  int64  
 1   Surname          9998 non-null   object 
 2   CreditScore      10001 non-null  int64  
 3   Geography        10001 non-null  object 
 4   Gender           10001 non-null  object 
 5   Age              9998 non-null   float64
 6   Tenure           10001 non-null  int64  
 7   EstimatedSalary  10001 non-null  object 
dtypes: float64(1), int64(3), object(4)
memory usage: 625.2+ KB

[ ]
churn_acct_info = pd.read_excel("/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/Bank_Churn_Messy.xlsx", sheet_name = 1)

churn_acct_info.head()


[ ]
churn_acct_info = pd.read_excel("Bank_Churn_Messy.xlsx", sheet_name =1)
churn_acct_info.head()


[ ]
churn_acct_info.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10002 entries, 0 to 10001
Data columns (total 7 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   CustomerId      10002 non-null  int64 
 1   Balance         10002 non-null  object
 2   NumOfProducts   10002 non-null  int64 
 3   HasCrCard       10002 non-null  object
 4   Tenure          10002 non-null  int64 
 5   IsActiveMember  10002 non-null  object
 6   Exited          10002 non-null  int64 
dtypes: int64(4), object(3)
memory usage: 547.1+ KB
Here, the "Exited" variable will be our variable of interest, i.e.,response variable for the Classification Model.


[ ]
churn_df = churn_cust_info.merge(churn_acct_info, how = "left", on = "CustomerId")

#Suppose, in the "churn_acct_info" dataset the "CustomerId" is named differently, say, "CustID"
#churn_df = churn_cust_info.merge(churn_acct_info, how="left", left_on="CustomerId", right_on="CustID")

churn_df.head()

Check for Duplicate Rows and Duplicate Columns


[ ]
churn_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10004 entries, 0 to 10003
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10004 non-null  int64  
 1   Surname          10001 non-null  object 
 2   CreditScore      10004 non-null  int64  
 3   Geography        10004 non-null  object 
 4   Gender           10004 non-null  object 
 5   Age              10001 non-null  float64
 6   Tenure_x         10004 non-null  int64  
 7   EstimatedSalary  10004 non-null  object 
 8   Balance          10004 non-null  object 
 9   NumOfProducts    10004 non-null  int64  
 10  HasCrCard        10004 non-null  object 
 11  Tenure_y         10004 non-null  int64  
 12  IsActiveMember   10004 non-null  object 
 13  Exited           10004 non-null  int64  
dtypes: float64(1), int64(6), object(7)
memory usage: 1.1+ MB

[ ]
churn_df = churn_df.drop("Tenure_y", axis = 1).rename({"Tenure_x":"Tenure"}, axis = 1)

churn_df.head()


[ ]
churn_df.duplicated(keep= False)


[ ]
churn_df.tail()


[ ]
churn_df = churn_df.drop_duplicates()

#To drop duplicates based only on the CustomerId column:
#churn_df = churn_df.drop_duplicates(subset="CustomerId")
#If I want to take the combination of CustomerId and Surname
#churn_df = churn_df.drop_duplicates(subset=["CustomerId", "Surname"])

churn_df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 0 to 10000
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10000 non-null  int64  
 1   Surname          9997 non-null   object 
 2   CreditScore      10000 non-null  int64  
 3   Geography        10000 non-null  object 
 4   Gender           10000 non-null  object 
 5   Age              9997 non-null   float64
 6   Tenure           10000 non-null  int64  
 7   EstimatedSalary  10000 non-null  object 
 8   Balance          10000 non-null  object 
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  object 
 11  IsActiveMember   10000 non-null  object 
 12  Exited           10000 non-null  int64  
dtypes: float64(1), int64(5), object(7)
memory usage: 1.1+ MB
Objective 2: Data Cleaning
Our Second Objective is to Clean the Data by fixing inconsistencies in labeling, handling erroneous values, and fixing currency fields.


[ ]
churn_df.head()


[ ]
# churn_df["EstimatedSalary"].str.strip("€").astype("float")
# Here, strip won't work as it only strip at the beginning and end. The negative sign creates a problem here.

churn_df["EstimatedSalary"] = churn_df["EstimatedSalary"].str.replace("€", "").astype("float")

churn_df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 0 to 10000
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10000 non-null  int64  
 1   Surname          9997 non-null   object 
 2   CreditScore      10000 non-null  int64  
 3   Geography        10000 non-null  object 
 4   Gender           10000 non-null  object 
 5   Age              9997 non-null   float64
 6   Tenure           10000 non-null  int64  
 7   EstimatedSalary  10000 non-null  float64
 8   Balance          10000 non-null  object 
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  object 
 11  IsActiveMember   10000 non-null  object 
 12  Exited           10000 non-null  int64  
dtypes: float64(2), int64(5), object(6)
memory usage: 1.1+ MB

[ ]
churn_df["Balance"] = churn_df["Balance"].str.replace("€", "").astype("float")

churn_df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 0 to 10000
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10000 non-null  int64  
 1   Surname          9997 non-null   object 
 2   CreditScore      10000 non-null  int64  
 3   Geography        10000 non-null  object 
 4   Gender           10000 non-null  object 
 5   Age              9997 non-null   float64
 6   Tenure           10000 non-null  int64  
 7   EstimatedSalary  10000 non-null  float64
 8   Balance          10000 non-null  float64
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  object 
 11  IsActiveMember   10000 non-null  object 
 12  Exited           10000 non-null  int64  
dtypes: float64(3), int64(5), object(5)
memory usage: 1.1+ MB

[ ]
churn_df.Exited.isnull()


[ ]

Start coding or generate with AI.

[ ]
churn_df.isnull().sum()


[ ]
# Count the number of Missing Values in each Column
missing_values_count = churn_df.isnull().sum()

# Total number of missing values in the DataFrame
total_missing = missing_values_count.sum()

print(f"Total missing values in the DataFrame: {total_missing}")
missing_values_count


[ ]
churn_df[churn_df.isna().any(axis=1) == True]


[ ]
churn_df = churn_df.fillna(value = {"Surname":"Missing", "Age": churn_df["Age"].median()})

churn_df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10000 entries, 0 to 10000
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   CustomerId       10000 non-null  int64  
 1   Surname          10000 non-null  object 
 2   CreditScore      10000 non-null  int64  
 3   Geography        10000 non-null  object 
 4   Gender           10000 non-null  object 
 5   Age              10000 non-null  float64
 6   Tenure           10000 non-null  int64  
 7   EstimatedSalary  10000 non-null  float64
 8   Balance          10000 non-null  float64
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  object 
 11  IsActiveMember   10000 non-null  object 
 12  Exited           10000 non-null  int64  
dtypes: float64(3), int64(5), object(5)
memory usage: 1.1+ MB

[ ]
churn_df.iloc[[28, 121, 9389]]


[ ]
churn_df.describe()


[ ]
churn_df["EstimatedSalary"] = churn_df["EstimatedSalary"].replace(-999999, churn_df["EstimatedSalary"].median())

churn_df.describe()


[ ]
churn_df.head()


[ ]
churn_df["Geography"].value_counts()


[ ]
#Easy way
## churn_df["Geography"] = churn_df["Geography"].replace({"FRA": "France", "Frence" : "France"})
churn_df["Geography"] = np.where(churn_df["Geography"].isin(["FRA", "France", "French"]), "France", churn_df["Geography"])

churn_df["Geography"].value_counts()

Wrapping up the Data Cleaning Steps in a Single Block of Code

[ ]
#Importing the Required Libraries   if not loaded yet
import pandas as pd
import numpy as np

churn_df = (
    pd.read_excel("/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/Bank_Churn_Messy.xlsx")
    .merge(pd.read_excel("/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/Bank_Churn_Messy.xlsx", sheet_name=1), how = "left", on = "CustomerId")
    .drop_duplicates()
    .drop("Tenure_y", axis = 1)
    .rename({"Tenure_x": "Tenure"}, axis = 1)
    .assign(
        EstimatedSalary = lambda x: x["EstimatedSalary"].str.replace("€", "").astype("float"),
        Balance = lambda x: x["Balance"].str.replace("€", "").astype("float"),
        Geography = lambda x: np.where(x["Geography"].isin(["FRA", "France", "French"]), "France", x["Geography"])
    )
    .assign(EstimatedSalary = lambda x: x["EstimatedSalary"].replace(-999999, x["EstimatedSalary"].median()))
)

churn_df = churn_df.fillna(value = {"Surname": "Missing", "Age": churn_df["Age"].median()})

churn_df.head()


Objective 3: Exploring the Data
Our third objective is to explore the target variable (Exited) and look at feature-target relationships for categorical and numeric fields:

Building a bar chart displaying the count of churners (Exited=1) vs. non-churners (Exited=0).

Exploring the categorical variables vs. the target, and look at the percentage of Churners by “Geography” and “Gender”.

Building box plots for each numeric field, broken out by churners vs. non-churners.

Building histograms for each numeric field, broken out by churners vs. non-churners.


[ ]
churn_df["Exited"].value_counts()


[ ]
churn_df["Exited"].value_counts().plot.bar()


[ ]
churn_df["Exited"].value_counts(normalize = True).plot.bar() #To see it in percentage


[ ]
import seaborn as sns

sns.barplot(data = churn_df, x="Geography", y = "Exited")


[ ]
import matplotlib.pyplot as plt

for col in churn_df.drop("Surname", axis = 1).select_dtypes("object"):
  sns.barplot(data=churn_df, x = col, y= "Exited")
  plt.show()


[ ]
sns.boxplot(data = churn_df, y = "Age", hue = "Exited")


[ ]
for col in churn_df.drop(["CustomerId", "Exited"], axis = 1).select_dtypes("number"):
  sns.boxplot(data=churn_df, y = col, hue= "Exited")
  plt.show()


[ ]
for col in churn_df.drop(["CustomerId", "Exited"], axis = 1).select_dtypes("number"):
  sns.histplot(data=churn_df, x = col, hue= "Exited", kde = True)
  plt.show()

Objective 4: Preparing the Data for Modelling
Our final objective is to prepare the data for modeling through feature selection, feature engineering, and data splitting:

Creating a new dataset that excludes any columns that aren’t be suitable for modeling.

Creating Dummy Variables for categorical fields.

Create a new “balance_v_income” feature, which divides a customer’s bank balance by their estimated salary, then visualize that feature vs. churn status.


[ ]
churn_df.head()


[ ]
modelling_df = churn_df.drop(["CustomerId", "Surname"], axis = 1)

modelling_df


[ ]
modelling_df = pd.get_dummies(modelling_df, drop_first = True, dtype = "int")

modelling_df.head()


[ ]
modelling_df["Balance_v_Sal"] = modelling_df["Balance"] / modelling_df["EstimatedSalary"]

modelling_df.head()


[ ]
sns.boxplot(data = modelling_df, y = "Balance_v_Sal")


[ ]
modelling_df.describe()


[ ]
sns.boxplot(data = modelling_df.query("Balance_v_Sal < 10"), y = "Balance_v_Sal")


[ ]
sns.barplot(data = modelling_df.query("Balance_v_Sal < 10"), y = "Balance_v_Sal", hue = "Exited")

Now, by Applying Machine Learning Algorithms we will identify which variables have impact on our Target Variable (Exited).

Before, that let's save the "modelling_df" dataframe in our Google Drive as a CSV file for further use.


[ ]
# Specify the path in your Google Drive
# file_path = '/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/modelling_df.csv'

# # Save the DataFrame as a CSV file
# modelling_df.to_csv(file_path, index=False)  # Set index=False to avoid saving the index as a column

# print(f"File saved to {file_path}")
File saved to /content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/modelling_df.csv
Bank Churn Customer Analysis: Classification
|| Building a Classification Model to Predict which bank customers are most likely to Churn ||

Objective 1: Checking Multicollinearity of the Data

[ ]
# Suppose it's a different Session and you need to load the saved "modelling_df" data again
# import pandas as pd
# modelling_df = pd.read_csv("/content/drive/MyDrive/Data Analytics & BI Career Path/Batch 2/Python/Bank Churn Customer Project/modelling_df.csv")

# modelling_df.head()

[ ]
#Let's First Inspect Our Dataframe
modelling_df.head()


[ ]
#Let's check if there's any potential multicollinearity through pairplot
#sns.pairplot(modelling_df.select_dtypes(include=['number'])) #That's with all the variables

# Create the pairplot for numeric variables only
numeric_cols = modelling_df.select_dtypes(include=['number']).columns
sns.pairplot(modelling_df[numeric_cols])
plt.show()

#Check if we can find any pattern between two features


[ ]
sns.heatmap(modelling_df.corr(numeric_only = True), vmin = -1, vmax =1, cmap = "coolwarm")

Objective 2: More Feature Engineering & Train-Test Split
Our second objective is to prepare the data for modeling through feature selection, feature engineering, and data splitting.

Create a new column, “income_v_products”, by dividing “EstimatedSalary” by “NumOfProducts”.

Split the data into train and test sets, with 20% of the rows in the test set.


[ ]
modelling_df["income_v_product"] = modelling_df["EstimatedSalary"] / modelling_df["NumOfProducts"]
modelling_df.head()


[ ]
sns.boxplot(modelling_df, x = "Exited", y= "income_v_product")


[ ]
# Now, we will spilt our data into Train and Test Set

from sklearn.model_selection import train_test_split
X = modelling_df.drop("Exited", axis = 1)
y = modelling_df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.info()
<class 'pandas.core.frame.DataFrame'>
Index: 8000 entries, 9255 to 7271
Data columns (total 13 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   CreditScore         8000 non-null   int64  
 1   Age                 8000 non-null   float64
 2   Tenure              8000 non-null   int64  
 3   EstimatedSalary     8000 non-null   float64
 4   Balance             8000 non-null   float64
 5   NumOfProducts       8000 non-null   int64  
 6   Geography_Germany   8000 non-null   int64  
 7   Geography_Spain     8000 non-null   int64  
 8   Gender_Male         8000 non-null   int64  
 9   HasCrCard_Yes       8000 non-null   int64  
 10  IsActiveMember_Yes  8000 non-null   int64  
 11  Balance_v_Sal       8000 non-null   float64
 12  income_v_product    8000 non-null   float64
dtypes: float64(5), int64(8)
memory usage: 875.0 KB

[ ]
X_test.info()
<class 'pandas.core.frame.DataFrame'>
Index: 2000 entries, 6253 to 6930
Data columns (total 13 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   CreditScore         2000 non-null   int64  
 1   Age                 2000 non-null   float64
 2   Tenure              2000 non-null   int64  
 3   EstimatedSalary     2000 non-null   float64
 4   Balance             2000 non-null   float64
 5   NumOfProducts       2000 non-null   int64  
 6   Geography_Germany   2000 non-null   int64  
 7   Geography_Spain     2000 non-null   int64  
 8   Gender_Male         2000 non-null   int64  
 9   HasCrCard_Yes       2000 non-null   int64  
 10  IsActiveMember_Yes  2000 non-null   int64  
 11  Balance_v_Sal       2000 non-null   float64
 12  income_v_product    2000 non-null   float64
dtypes: float64(5), int64(8)
memory usage: 218.8 KB
Objective 3: Build & Evaluate a Logistic Regression Model
Our third objective is to Fit a Logistic Regression Model and Evaluate it by using a Confusion Matrix, ROC Curve, and Precision & Recall:

Fit a logistic regression model on our training data.

Build a confusion matrix to evaluate our model.

Calculate accuracy, precision, recall, and F1 for our test data.

Plot an ROC curve and calculate the AUC statistic.


[ ]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve

[ ]
logreg = LogisticRegression()

lr = logreg.fit(X_train, y_train)

print(f"Train Accuracy: {lr.score(X_train, y_train)}")
print(f"Test Accuracy: {lr.score(X_test, y_test)}")
Train Accuracy: 0.787625
Test Accuracy: 0.802
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

[ ]
logreg = LogisticRegression(max_iter = 100000)

lr = logreg.fit(X_train, y_train)

print(f"Train Accuracy: {lr.score(X_train, y_train)}")
print(f"Test Accuracy: {lr.score(X_test, y_test)}")
Train Accuracy: 0.808125
Test Accuracy: 0.817
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
See, the accuracy has increased!


[ ]
confusion_matrix(y_train, lr.predict(X_train))
array([[6127,  229],
       [1306,  338]])

[ ]
print(f"Train Accuracy: {lr.score(X_train, y_train)}")
Train Accuracy: 0.808125
Precision is the proportion of all the model's positive classifications that are actually positive. It is mathematically defined as:

Precision = TP / (TP + FP)

TP = True positives

FP = False Positives


[ ]
precision_score(y_train, lr.predict(X_train))
0.5961199294532628
A recall score is a metric that measures how well a machine learning model identifies positive instances in a dataset. It's calculated by dividing the number of true positives by the total number of positive samples. Recall tells you what proportion of all the positive cases in your data did your model correctly predict:

Recall = TP / (TP + FN)

TP = True positives

FN = False negatives

image.png

https://www.kdnuggets.com/2022/11/confusion-matrix-precision-recall-explained.html


[ ]
recall_score(y_train, lr.predict(X_train))
0.20559610705596107

[ ]
#Let''s check the Coefficients of the Model to find which Features have how much impact on our Target Variable
list(zip(X_train, lr.coef_[0]))
[('CreditScore', np.float64(-0.0020469841909369225)),
 ('Age', np.float64(0.06021977990669507)),
 ('Tenure', np.float64(-0.020629551226091114)),
 ('EstimatedSalary', np.float64(-1.2756363220244559e-05)),
 ('Balance', np.float64(1.0800240928518866e-06)),
 ('NumOfProducts', np.float64(0.4106706467033999)),
 ('Geography_Germany', np.float64(0.9352756423022984)),
 ('Geography_Spain', np.float64(0.18052587480896165)),
 ('Gender_Male', np.float64(-0.7329973219151157)),
 ('HasCrCard_Yes', np.float64(-0.33030207295081765)),
 ('IsActiveMember_Yes', np.float64(-0.33030207295081765)),
 ('Balance_v_Sal', np.float64(0.0019811388395756416)),
 ('income_v_product', np.float64(1.614573452591591e-05))]

[ ]
# Let's calculate the Accuracy Metrics for Test Data
confusion_matrix(y_test, lr.predict(X_test))
array([[1551,   56],
       [ 310,   83]])

[ ]
print(f"Test Accuracy: {lr.score(X_test, y_test)}")
Test Accuracy: 0.817
Accuracy didn't drop; hence, no over-fitting.


[ ]
precision_score(y_test, lr.predict(X_test))
0.5971223021582733

[ ]
recall_score(y_test, lr.predict(X_test))
0.21119592875318066
Didn't drop much.

image.png


[ ]
f1_score(y_test, lr.predict(X_test))
0.31203007518796994
image.png

Area under the curve (AUC):

The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative.

The perfect model above, containing a square with sides of length 1, has an area under the curve (AUC) of 1.0. This means there is a 100% probability that the model will correctly rank a randomly chosen positive example higher than a randomly chosen negative example.

In more concrete terms, a spam classifier with AUC of 1.0 always assigns a random spam email a higher probability of being spam than a random legitimate email. The actual classification of each email depends on the threshold that you choose.


[ ]
y_probs = lr.predict_proba(X_test)[:, 1]
fpr1, tpr1, tresholds = roc_curve(y_test, y_probs)
auc_score = auc(fpr1, tpr1)

# Plot the ROC Curve
plt.plot(fpr1, tpr1, label = f'LR(AUC = {auc_score:.2f})')

# Draw Random Guess
plt.plot([0,1], [0,1], 'k--', label = 'Random Guess (AUC = 0.50)')

# Modify Formatting
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Bank Churn Model')
plt.legend()
plt.show()


Objective 4: Fit & Tune a Random Forest Model
Our final objective is to fit a random forest model, tune it using cross validation, and evaluate test accuracy, AUC score, and feature importance:

Fit a random forest model with default hyperparameters.

Use cross validation to tune your model's hyperparameters.

Report the final test accuracy and AUC score.

Build a bar chart that shows feature importance.


[ ]
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf = rf.fit(X_train, y_train)

print(f"Train Accuracy: {rf.score(X_train, y_train)}")
print(f"Test Accuracy: {rf.score(X_test, y_test)}")
Train Accuracy: 1.0
Test Accuracy: 0.865
Random Forest is extremely overfit.

Hyperparameter Tuning:


[ ]
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state = 24, n_jobs = -1)

params = {
    'n_estimators': [100, 300, 500, 700, 800],
    'max_features': ["sqrt"],
    'bootstrap': [True],
    'max_samples': [None, 0.5],
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [5, 10, 20],
}

grid = RandomizedSearchCV(
    rf,
    params,
    n_iter = 50,
    scoring = "accuracy"
)

grid.fit(X_train, y_train)

grid.best_params_
{'n_estimators': 100,
 'min_samples_leaf': 10,
 'max_samples': None,
 'max_features': 'sqrt',
 'max_depth': 9,
 'bootstrap': True}

[ ]
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': np.arange(start = 700, stop = 1000, step = 100),
    'min_samples_leaf': [8, 10, 12],
    'max_samples': [None, .3],
    'max_features': ["sqrt"],
    'max_depth': np.arange(start = 8, stop = 10, step = 1),
    'bootstrap': [True]
}

grid = GridSearchCV(
    rf,
    params,
    scoring = "accuracy"
)

grid.fit(X_train, y_train)

grid.best_params_
{'bootstrap': True,
 'max_depth': 9,
 'max_features': 'sqrt',
 'max_samples': None,
 'min_samples_leaf': 10,
 'n_estimators': 900}

[ ]
rf = RandomForestClassifier(**{
    'n_estimators': 900,
    'min_samples_leaf': 10,
    'max_samples': None,
    'max_features': "sqrt",
    'max_depth': 9,
    'bootstrap': True}
)

rf = rf.fit(X_train, y_train)

print(f"Train Accuracy: {rf.score(X_train, y_train)}")
print(f"Test Accuracy: {rf.score(X_test, y_test)}")
Train Accuracy: 0.87425
Test Accuracy: 0.8615

[ ]
y_probs = rf.predict_proba(X_test)[:, 1]
fpr1, tpr1, tresholds = roc_curve(y_test, y_probs)
auc_score = auc(fpr1, tpr1)

auc_score
0.8697524032105087
Feature Importance:


[ ]
importance = pd.DataFrame(
    {"Features": X_train.columns,
     "Importance": rf.feature_importances_}
).sort_values(by = "Importance", ascending = False).iloc[:20]

sns.barplot(x = "Importance", y = "Features", data = importance)

So, Age is the most influential Feature; Apart from that Number of Products, Balance, Geography of Germany, Active Member and some of our Engineered Features were found Important.

While Gender, Tenure, Credit Score do not impact Churning Status that much.

Colab paid products - Cancel contracts here
