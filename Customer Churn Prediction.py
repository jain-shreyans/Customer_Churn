#!/usr/bin/env python
# coding: utf-8

# <h1><center>Customer Churn Modeling </center></h1>

# ***

# # TABLE OF CONTENTS
# 

# + Introduction
# + Objective
# + Problem Statement
# + Approach
# + EDA, Model & Results
# + Conclusion
# + References

# <hr>

# ### Introduction
# 
# A market has two key Parties
# + Buyer
# + Seller
# 
# Often the sellers have a profit motive and buyers an aim to maximize utility.
# 
# + In midst of this cycle  many of the buyers change their habbits, switch to different products, due to pricing change, income or other factors. Thus for a business owner (seller) it is cruical to know whether a customer wil stay loyal in terms of buying or not. This is other terms is known as **customer churn**. Wherein a customer stops purchasing from the the particular company / organization. 
# 
# 

# <hr>

# ### Objectives
# 
# Predict the customers who are going to churn soon with resonable accuracy. 
# - Utlimately helping in making business decisions
# 
# 

# <hr>

# ### Problem Definition
# 
# Financial institutions have clients those who close accounts or switch to different institutions.
# This impacts the revenue model of the institution.
# Thus from a business point of view, it is cruical to predict whether a customer will churn or not. 
# 
# In this context a customer having closed all their active accounts with the bank is said to have churned.
# 
# **" NOTE : "** Churn can mean something else as well basedd on the scenario. Eg: customer not trasacting for more than a year can be treated as churned. 
# 
# #### PROBLEM : Predict customers who are going to churn soon

# <hr>

# ### Approach
# This is an extension to the Problem Defnintion. You have to mention the process/appraoch that you have followed in order to reach out the above problem defintion.

# <hr>

# In[410]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[411]:


# Reading the dataset
df = pd.read_csv("D:\Machine_Learning\customer_churn\churn.csv")
df.head(15) 


# In[412]:


df.shape


# Ten Thousand rows and 14 columns

# In[413]:


df_numeric = df._get_numeric_data()
df_numeric.describe()


# In[414]:


df.describe(include = ['O']) # Describes categorical columns


# ### EDA

# In[415]:


# unique customers 
df.CustomerId.nunique()


# In[416]:


df["Exited"].value_counts()
# Proportion of churned and not churned is high, which might affect the training model as data is imbalanced. 


# In[417]:


sns.boxplot(y=df['CreditScore'])


# In[418]:


df.groupby(['Geography']).agg({'RowNumber':'count', 'Exited':'mean'})


# In[419]:


# Frace has the most customers who have churned, but the average number of churned customer in Germany is higher.


# In[420]:


sns.countplot(x= 'Gender', data = df)


# In[421]:


sns.boxplot(y=df['Age'])


# In[422]:


sns.boxplot(y=df['Tenure'])


# In[423]:


sns.boxplot(y=df['Balance'])


# In[424]:


sns.countplot(x = 'NumOfProducts', data = df)


# In[425]:


#Most of the customers have 1 or 2 products. 80/20 Rule.


# In[426]:


df.groupby(['NumOfProducts']).agg({'RowNumber':'count', 'Exited':'mean'}) 


# In[427]:


# People who bought 4 products always churn, moreover ones who have 3 products exit 83% of the time


# In[428]:


sns.countplot(x = 'HasCrCard', data = df)


# In[429]:


sns.countplot(x = 'IsActiveMember', data = df)


# In[430]:


sns.boxplot(y=df['EstimatedSalary'])


# In[431]:


# Dropping columns that are useful for modeling
df1 = df.drop(columns = ['RowNumber', 'CustomerId'])


# In[432]:


df1_numeric = df1._get_numeric_data()


# In[433]:


corr = df1_numeric.corr()
plt.figure(figsize = (15,8))
sns.heatmap(corr, cmap = 'Blues', annot = True)


# ### Label Enconding for categorical Data

# In[434]:


# Separating out different columns into various categories as defined above
target_var = ['Exited']

# numerical columns
num_feats = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# categorical columns
cat_feats = [ 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember']


# In[435]:


# Tenure, NumOfProducts has order value.
# HasCrCard, IsActiveNumber, Gender is binary categorical


# In[436]:


# label encoding With  the sklearn method
le = LabelEncoder()

# Label encoding of categorical variable
df1['Gender'] = le.fit_transform(df1['Gender'])
df1['Geography'] = le.fit_transform(df1['Geography'])


# In[437]:


df1['bal_per_product'] = df1.Balance/(df1.NumOfProducts)
df1['bal_by_est_salary'] = df1.Balance/(df1.EstimatedSalary)
df1['tenure_age_ratio'] = df1.Tenure/(df1.Age)
new_cols = ['bal_per_product', 'bal_by_est_salary', 'tenure_age_ratio']
# Ensuring that the new column doesn't have any missing values
df1[new_cols].isnull().sum()


# In[438]:


df1.head()


# In[439]:


df2 = df1[new_cols]
df2['Age'] = df1['Age']
df2['Exited'] = df1['Exited']
df2.head()


# In[440]:


sns.heatmap(df2.corr(), annot=True)


# ### Split dataset into train and test

# In[441]:


X = df1.drop(['Exited'], axis = 1) # axis for column
y = df['Exited']


# In[442]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# ## ALGORITHM / MODEL

# ### DECISION TREE

# In[485]:


dec_tree = DecisionTreeClassifier(criterion='gini',max_depth =5, random_state = 42)
dec_tree.fit(X_train,y_train)
print("Decision Tree Classification Score: ",dec_tree.score(X_test,y_test))


# In[489]:


dec_tree.tree_.max_depth


# ### Hyperparameter = max depth and criterion  to measure impurity (gini, entropy) 

# In[483]:


for i in range(1,25):
    dec_tree = DecisionTreeClassifier(criterion='gini',max_depth =i ,random_state = 42)
    dec_tree.fit(X_train,y_train)
    print("Tree score with depth",i,"=",dec_tree.score(X_test,y_test) )


# In[446]:


#tree.export_graphviz(dec_tree,out_file="tree.dot",class_names=["Did not churn", "churned"], filled = True)


# In[486]:


y_pred = dec_tree.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)
sns.heatmap(conf_mat,annot=True,fmt=".0f")


# In[487]:


print(classification_report(y_test, y_pred))


# In[488]:


# Accuracy is 86%


# ### LOGISTIC REGRESSION

# In[450]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[451]:


lr = LogisticRegression(random_state = 42)
lr.fit(X1_train, y1_train)


# In[452]:


y1_pred = lr.predict(X1_test)
conf_mat = confusion_matrix(y1_test,y1_pred)
sns.heatmap(conf_mat,annot=True,fmt=".0f")


# In[453]:


print(classification_report(y_test, y1_pred))


# In[454]:


# Accuracy is 80%


# ### Hyperparameter = penalty for wrong predictions

# In[455]:


# only these two are used as lbfgs solver supports only 'l2' or 'none' penalties
penalty = ['l2','none']
for i in penalty:
    lr = LogisticRegression(random_state = 42, penalty = i)
    lr.fit(X1_train, y1_train)
    print(i,"penalty has a mean accuracy of ", lr.score(X, y))


# In[456]:


# Seems to have no difference.


# ## Better amongst Decision tree and Logistic Regression to predict churn is DECISION TREE 

# In[457]:


# take user input 


# In[458]:


df1.columns


# In[461]:


l = []
l.append(input("Enter your Credit Score:"))
l.append(input("Enter your Country ():"))
l.append(input("Gender:"))
age = input("Enter your Age:")
l.append(age)
t = input("Enter Tenure:")
l.append(t)
b = input("Enter your Balance:")
l.append(b)
n = input("Enter your Num Of Products:")
l.append(n)
l.append(input("Enter your HasCrCard:"))
l.append(input("Enter your IsActiveMember:"))
es = input("Enter your EstimatedSalary:")
l.append(int(es))
l.append(int(b)/int(n))
l.append(int(b)/int(es))
l.append(int(t)/int(age))


# In[462]:


output = dec_tree.predict([l])
if output ==0:
    print(" Customer will not churn with an accuracy of 86%")
else: 
    print("Customer will churn with an accuracy of 86%")


# <hr/>

# ### References
# 
# 1. https://www.thepythoncode.com/article/customer-churn-detection-using-sklearn-in-python#intro
# 1. https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b
# 1.https://scikit-learn.org/
# 
# 
