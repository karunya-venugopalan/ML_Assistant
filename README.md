# PA_Assistant
The project uses python to analyse and predict any training dataset such as Hoam loan prediction, Diabetes prediction etc. It fits the training data into different Supervised Machine Learning models and chooses the one with highest accuracy to make predictions. Graphs and inferences from the analysis will be written and stored in a pdf file. 

Program file:
home_loan_prediction.py - Main python file which performs the analysis and prediction

The files in Input files are as follows:
1. parameters_file.csv - Format of this file should be: 
                            "Columns"
                             *Column name of the decision variable in your train data*
                             *Count of the number of categorical variables in train which you would like to consider for prediction. Let the count be n*
                             *Next n lines: Column names of the n categorical variables*
                             *Count of the number of numerical variables in train which you would like to consider for prediction. Let the count be m*
                             *Next m lines: Column names of the m numerical variables*

2. train.csv - Train data set which has decision variable
3. test.csv - Test data which contains all the columns of train.csv except the decision variable which we should predict

The files in Output files are as follows:
1. TextPdf.pdf - Consists of the text part of the Analysis process
2. plots.pdf - Consists of the plots done in the Analysis process. A few sample screenshots are given below:

<img width="327" alt="graph1" src="https://user-images.githubusercontent.com/57533513/102979146-a9d2b200-452b-11eb-93a2-58da19a86dde.PNG">


3. combined_output.pdf - Combination of the above two files
4. test_copy.csv - copy of the test.csv which was provided along with a new column with the prediction made by the program

