import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# Loading train data file using pandas library
data_train = pd.read_csv("data_file.csv")

# get unique practice Ids
practiceId = data_train.id.unique()

# Check practice shape
print(practiceId.shape)

# declared some required arrays for test features
id = []
month = []
visits = []
no_of_appts = []
production = []


# Start the loop by practice to generate 2021 visits and no_of_appts
warnings.filterwarnings("ignore")
for pracId in practiceId:
    newdf = data_train[(data_train.id == pracId)]

    Jandf = newdf[(newdf.month == 1)]
    id.append(pracId)
    month.append(1)
    visits.append(round(Jandf.visits.mean()))
    production.append(round(Jandf.production.mean(),2))

    Febdf = newdf[(newdf.month == 2)]
    id.append(pracId)
    month.append(2)
    visits.append(round(Febdf.visits.mean()))
    production.append(round(Febdf.production.mean(),2))

    Mardf = newdf[(newdf.month == 3)]
    id.append(pracId)
    month.append(3)
    visits.append(round(Mardf.visits.mean()))
    production.append(round(Mardf.production.mean(),2))

    Aprdf = newdf[(newdf.month == 4)]
    id.append(pracId)
    month.append(4)
    visits.append(round(Aprdf.visits.mean()))
    production.append(round(Aprdf.production.mean(),2))

    
# Dictionary of lists 
##dict = {'id': id, 'month': month, 'year': "2021", 'visits' : visits, 'no_of_appts' : no_of_appts, 'production' : production}
dict = {'id': id, 'month': month, 'year': "2021", 'visits' : visits, 'production' : production}
dp = pd.DataFrame(dict)

# Saving the dataframe to test.csv
dp.to_csv('test.csv', index=False)

# Loading test data file using pandas library
data_test = pd.read_csv("test.csv")

# Check if there is any column with null value
print(data_train.isnull().sum())

# Check whether the duplicate exist or not in train csv file
print(data_train.duplicated().sum())

# Check train data csv file shape
print(data_train.shape)

# Check train data csv file information
print(data_train.info())

# Display sample 10 records
print(data_train.sample(n=10))

# Check count, mean, average, max, min and other information
print(data_train.describe().transpose())

# Show 10 sample records
print(data_train.sample(n=10))

# Declared some required arrays for result file

countVal = 0
id = []
month = []
year = []
pred_prod = []


# Defining MAPE function
def MAPE(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

# Start the loop by practice to generate 2021 visits and no_of_appts
warnings.filterwarnings("ignore")

for pracId in practiceId:

    # Filter the Train and Test data set by practice Id
    new_data_train = data_train[(data_train.id == pracId)]
    new_data_test = data_test[(data_test.id == pracId)]

    # Drop practice id column, this is not an important feature
    new_data_train.drop(['id'], axis=1, inplace=True)
    new_data_test.drop(['id'], axis=1, inplace=True)

    # Separating the dependent and independent data variables into two data frames.
    X_train = new_data_train.drop(['production'], axis=1)
    X_train = X_train.drop(['no_of_appts'], axis=1)

    X_test = new_data_test.drop(['production'], axis=1)

    # Set target value
    y_train = new_data_train['production']
    y_test = new_data_test['production']

    # Covert it to Arrays
    X_train = X_train.values
    y_train = y_train.values

    X_test = X_test.values
    y_test = y_test.values

    # Splitting the Train and Test data with 90% for Training, and 10% for Testing
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    # Set model tuning parameters
    model = SVR(kernel='linear', C=100, gamma=0.001)

    # fit model for training
    model.fit(X_train, y_train)

    # predict the values on test set
    y_pred = model.predict(X_test)

    # mean-absolute-percentage-error calculation
    mape = MAPE(y_test,y_pred)


# result.csv

    for test in X_test:
        month.append(test[0])
        year.append(test[1])
        id.append(pracId)

    for pred in y_pred:
        pred_prod.append(round(pred,2))

##    if mape >5 and mape <=10:
    if mape <=10:
        countVal = countVal+1
        
    print("PracticeId: ", pracId ," MAPE: ", round(mape,2))
    accuracy = 100 - np.mean(mape)
    print('Model Accuracy:', round(accuracy, 2), '%.')

 
print("Practices under 15% Error: ",countVal)

# Create dictionary for storing result object 
dict = {'id': id, 'month': month, 'year': year, 'production' : pred_prod}
dp = pd.DataFrame(dict)

# Saving the dataframe to support_vector_result.csv file
dp.to_csv('support_vector_result.csv', index=False)
