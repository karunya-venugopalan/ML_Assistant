# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:32:18 2020

@author: Karunya V
"""
#Importing libraries
from docx import Document
from docx.shared import Inches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.compat import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import PyPDF2 
import pickle
warnings.filterwarnings("ignore")

#opening pdf file in which plots will be stored
pp = PdfPages('C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//plots.pdf')

#opening txt file to write model accuracies and prediction
text_file = open('C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//text_doc.txt',"a")

#opening pdf file to convvert the above txt file to pdf
text_pdf = FPDF()
text_pdf.add_page()
text_pdf.set_font("Arial", size = 14) 

#loading the train and test data
train = pd.read_csv("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Input files//train.csv")
test = pd.read_csv("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Input files//test.csv")
parametersFile = pd.read_csv("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Input files//parameters_file.csv")

#Making a copy of the data
train_original = train.copy()
test_original = test.copy()

#printing the datatypes
print(train.dtypes)

#Knowing the data
print('Training data: ',train.shape)
train.head()

print('Testing data: ',test.shape)
test.head()


#Storing the column names under the given variables:
#decisionVar - decision variable
decisionVar = parametersFile['Columns'][0]

#Catagorical variables
categorical_count = parametersFile['Columns'][1]
categorical=[]
for i in range(int(categorical_count)):
  categorical.append(parametersFile['Columns'][2+i])

#Numerical variables
numerical_count = parametersFile['Columns'][int(categorical_count)+2]
numerical=[]
for i in range(int(numerical_count)):
  numerical.append(parametersFile['Columns'][int(categorical_count)+3+i])
    
#Univariate analysis(target variable - loan status)
train[decisionVar].value_counts()

# Normalize can be set to True to print proportions instead of number 
train[decisionVar].value_counts(normalize=True)*100

#box plot for the normalised values 
train[decisionVar].value_counts(normalize=True).plot.bar(title = decisionVar)
plt.savefig(pp,format='pdf')


#Analysing categorical variables

#Calculating no.of rows for subplots
if(int(categorical_count)%2 == 0):
    c_rows = int(categorical_count)/2
else:
    c_rows = (int(categorical_count)/2)+1
    
j=1
fig = plt.figure(figsize=[8,8])
fig.suptitle("Analysing categorical variables")
fig.subplots_adjust(hspace=1.6, wspace=0.4)
for i in categorical:
  train[i].value_counts()
  plt.subplot(int(c_rows),2,j)
  j+=1
  fig = train[i].value_counts(normalize=True).plot.bar(title = i, figsize=(8,10))
plt.savefig(pp,format='pdf')


#Analysing numerical variables

#Calculating number of rows for subplots
if(int(numerical_count)%2 == 0):
    n_rows = int(numerical_count)/2
else:
    n_rows = (int(numerical_count)/2)+1
n_rows *= 2

j=1
fig = plt.figure(figsize=[6,10])
fig.suptitle("Analysing categorical variables")
fig.subplots_adjust(hspace=0.8, wspace=0.4)
for i in numerical:
  train[i].fillna(train[i].median(),inplace=True)
  plt.subplot(int(n_rows),2,j)
  sns.distplot(train[i]);

  plt.subplot(int(n_rows),2,j+1)
  train[i].plot.box(figsize=(8,10))
  j+=2
plt.savefig(pp,format='pdf')


#Categorical independent variable Vs Decision variable

for i in categorical:
  var = pd.crosstab(train[i],train[decisionVar])
  fig = var.div(var.sum(1).astype(float),axis=0).plot.bar(stacked=True, figsize = (5,6), title = i)
  plt.savefig(pp,format='pdf')

#Numerical Independent variables Vs Decision variable
fig = plt.figure(figsize=[6,12])
j=1
fig.suptitle("Analysing numerical variables vs decision variable")
fig.subplots_adjust(hspace=1.2)
for i in numerical:
    plt.subplot(int(n_rows),1,j)
    j+=1
    train.groupby(decisionVar)[i].mean().plot.bar(figsize = (5,10), title = i)
    plt.xlabel(i)
plt.savefig(pp,format='pdf')


#Replacing non-numerical decision variable values with numerical values
train[decisionVar].replace('N', 0,inplace=True)
train[decisionVar].replace('Y', 1,inplace=True)

#removing symbols(+,-) from columns for furthur analysis
for i in categorical:
    train[i] = train[i].astype(str).str.rstrip('+-')
    test[i] = test[i].astype(str).str.rstrip('+-')

#Visualising correlation between all the numerical values
matrix = train.corr()
f, ax = plt.subplots(figsize=(10, 13))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True)
plt.savefig(pp,format='pdf')

#Missing values

#Filling missing values of categorical variables with mode
for i in categorical:
    train[i].fillna(train[i].mode()[0],inplace=True)
    test[i].fillna(test[i].mode()[0],inplace=True)

#Filling missing values of numerical variables with median
for i in numerical:
    train[i].fillna(train[i].median(),inplace=True)
    test[i].fillna(test[i].median(),inplace=True)
    
#Model building
text_file.write("Model building:\n")

#List of algorithms being used to build models
model_list = ['logistic_model','tree_model','forest_model','xgb_model']
model_accuracy = []

#dropping the columns which are not used for model building
for col in train.columns:
    if col not in categorical and col not in numerical and col != decisionVar:
        train = train.drop(col,1)
        test = test.drop(col,1)

#skitit-learn(Sklearn) requires the target variable to be in a seperate dataset. So, removing it from train and naming the rest of it 'X' and storing loan status in 'y'

X=train.drop(decisionVar,1)
y=train[decisionVar]

#We see that the categorical variables have non-numerical values which can't be used in logistic regression. So we introduce 'dummies' to replace them with numerical values.

X = pd.get_dummies(X)
X.columns #Here, we see that columns are created for nan values which are not necessary. drop those columns?

X.drop(list(X.filter(regex = 'nan')), axis = 1, inplace = True)

train = pd.get_dummies(train)
train.columns
train.drop(list(train.filter(regex = 'nan')), axis = 1, inplace = True)
test = pd.get_dummies(test,dummy_na=False)
test.columns
test.drop(list(test.filter(regex = 'nan')), axis = 1, inplace = True)
X.columns

#Now we are supposed to build the regression model using the train data and test using test data.
#In order to validate the model, we divide the train data into train and validation. 70% of train as train and 30% as validation
from sklearn.model_selection import train_test_split   
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3,random_state=1)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(x_train,y_train)
with open('logistic_model.pickle','wb') as f:
    pickle.dump(logistic_model,f)

#Predicting the Validation set using the model
pred_cv_logistic=logistic_model.predict(x_cv)
pred_cv_logistic
score_logistic = accuracy_score(pred_cv_logistic,y_cv)*100 
text_file.write("Logistic regression score (logistic_model): ")
text_file.write(str(score_logistic))
max_accuracy = score_logistic
model_accuracy.append(score_logistic)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=1)
tree_model.fit(x_train,y_train)

pred_cv_tree=tree_model.predict(x_cv)
score_tree =accuracy_score(pred_cv_tree,y_cv)*100
text_file.write("\nDecision tree score (tree_model): ")
text_file.write(str(score_tree)) 
if(score_tree > max_accuracy):
    max_accuracy = score_tree
model_accuracy.append(score_tree)

#Random forest
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)
forest_model.fit(x_train,y_train)

pred_cv_forest=forest_model.predict(x_cv)
score_forest = accuracy_score(pred_cv_forest,y_cv)*100
text_file.write("\nRandom forest score (forest_model): ")
text_file.write(str(score_forest))
if(score_forest > max_accuracy):
    max_accuracy = score_forest
model_accuracy.append(score_forest)

#Finding important features that determine the loan status
f, ax = plt.subplots(figsize=(10, 13))
importances = pd.Series(forest_model.feature_importances_,index=X.columns)
importances.plot(kind='barh', figsize=(12,8))
plt.savefig(pp,format='pdf')
pp.close()

#XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=50,max_depth=4)
xgb_model.fit(x_train,y_train)

pred_xgb=xgb_model.predict(x_cv)
score_xgb = accuracy_score(pred_xgb,y_cv)*100
text_file.write("\nXGBoost score (xgb_model): ")
text_file.write(str(score_xgb))
if(score_xgb > max_accuracy):
    max_accuracy = score_xgb
model_accuracy.append(score_xgb)

best_model_index = model_accuracy.index(max(model_accuracy))
best_model = model_list[best_model_index]
text_file.write("\n\nModel with highest accuracy is: ")
text_file.write(str(best_model))
text_file.write("\n\nUsing this model to predict the test data: \n")

with open(best_model+'.pickle','rb') as f:
    model = pickle.load(f)
pred_test = model.predict(test)
text_file.write(str(pred_test))
text_file.close()

#Converting txt file to text_pdf
text_file = open("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//text_doc.txt","r")

for x in text_file: 
    text_pdf.cell(200, 10, txt = x, ln = 1, align = 'L') 

text_pdf.output("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//TextPdf.pdf")  
text_file.close()

#Combining plots and text_pdf
pdf1File = open('C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//TextPdf.pdf','rb')
pdf2File = open('C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//plots.pdf','rb')

pdf1Reader = PyPDF2.PdfFileReader(pdf1File)
pdf2Reader = PyPDF2.PdfFileReader(pdf2File)

pdfWriter = PyPDF2.PdfFileWriter()

for pageNum in range(pdf1Reader.numPages):
  pageObj = pdf1Reader.getPage(pageNum)
  pdfWriter.addPage(pageObj) 
  
for pageNum in range(pdf2Reader.numPages):
  pageObj = pdf2Reader.getPage(pageNum)
  pdfWriter.addPage(pageObj)

pdfOutputFile = open('C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//Output files//combined_output.pdf','wb')
pdfWriter.write(pdfOutputFile)
pdfOutputFile.close()
pdf1File.close()
pdf2File.close()

test_copy = pd.read_csv("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//input files//test.csv")
test_copy["Loan status"] = pred_test
test_copy.to_csv("C://Users//Karunya V//Documents//Projects//Prediction of Home loan in python//output files//test_copy.csv", index=False)