import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from keras import models, layers
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV


# Loading data file using pandas library
data_train = pd.read_csv("dataset/data_file.csv")
data_test = pd.read_csv("dataset/test.csv")

# Check if there is any column with null value
print("df: ",data_train.isnull().sum())

# check unique by id
practiceId = data_train.id.unique()
print(practiceId.shape)

# check csv file shape
print(data_train.shape)

# check csv file information
print(data_train.info())

# display sample 10 records
print(data_train.sample(n=10))

# check count, means, ag and other information
print(data_train.describe().transpose())
print(data_train.sample(n=10))

# declared some required variable and arrays
countVal = 0
id = []
month = []
year = []
pred_prod = []


#Defining MAPE function
def MAPE(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

# for loop by practice id
warnings.filterwarnings("ignore")
for pracId in practiceId:
    new_data_train = data_train[(data_train.id == pracId)]
    new_data_test = data_test[(data_test.id == pracId)]

    # check duplicate
    # print("duplicate: ", new_data_train.duplicated().sum())

    # drop practice id column
    new_data_train.drop(['id'], axis=1, inplace=True)
    new_data_test.drop(['id'], axis=1, inplace=True)

    #Separating the dependent and independent data variables into two data frames.
    X = new_data_train.drop(['production'], axis=1)
    X = X.drop(['no_of_appts'], axis=1)
    X = X.drop(['visits'], axis=1)

    X1 = new_data_test.drop(['production'], axis=1)
    X1 = X1.drop(['no_of_appts'], axis=1) 
    X1 = X1.drop(['visits'], axis=1)

    # set target value
    y = new_data_train['production']
    y1 = new_data_test['production']
    X = X.values
    y = y.values

    X1 = X1.values
    y1 = y1.values

##    pca=PCA(n_components=2)
##    X = pca.fit_transform(X)

    # Splitting the dataset into 80% training data and 20% testing data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=0)

##    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=0)

##    sc = StandardScaler()   
##    X_train = sc.fit_transform(X_train)
##    X_test = sc.fit_transform(X_test)
    
##    model = SVR(kernel='linear', C=100, gamma=0.01)
##    model.fit(X_train,y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

##    # defining parameter range
##    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
##                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
##                  'kernel': ['rbf', 'Sigmoid','Linear']} 
##      
##    grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
##      
##    # fitting the model for grid search
##    grid.fit(X_train, y_train)
##
##    print(grid.best_params_)
##  
##    # print how our model looks after hyper-parameter tuning
##    print(grid.best_estimator_)

    # TOP IS Linear Regression 230 result<=15 232, without field 140 

##    model = LinearRegression()
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

##    model = LinearRegression()
##    model.fit(X, y)
##    y_pred = model.predict(X1)
##    mape = MAPE(y1,y_pred)

##    print('Coefficients: \n', model.coef_)


# initial keras model
##    model = models.Sequential()
##    model.add(layers.Dense(2, activation='relu', input_shape=[X_train.shape[1]]))
##    model.add(layers.Dense(2, activation='relu'))
##
##    # output layer
##    model.add(layers.Dense(1))
##
##    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
##
##    print(model.evaluate(X_test, y_test))
##
##    # we get a sample data (the first 2 inputs from the training data)
##    # we call the predict method
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # 167
##    model = GradientBoostingRegressor(n_estimators=40)
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

# Gradient boosting Regression, it is good 164
##    model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=100, max_depth=3)
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # ada boost regressor

##    rng = np.random.RandomState(1)
##    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

      # 15% 242
##    model = LGBMRegressor(learning_rate=0.02,n_estimators=300,boosting_type='gbdt')
      # 15% 92
##    model = LGBMRegressor(learning_rate=0.02,n_estimators=300,boosting_type='dart')
      # 15% 260
##    model = LGBMRegressor(learning_rate=0.02,n_estimators=300,boosting_type='goss')

##    model = LGBMRegressor(learning_rate=0.01,n_estimators=300,boosting_type='goss')

      # 132 without field
##    model = LGBMRegressor(learning_rate=0.05,n_estimators=500,boosting_type='gbdt')
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)
			
##    model = linear_model.BayesianRidge()
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # get importance
##    importance = lr.coef_
    # summarize feature importance
##    for i,v in enumerate(importance):
##        print('Feature: %0d, Score: %.5f' % (i,v))
	

    # TOP IS Linear Regression 230 result<=15 197
##    model = DecisionTreeRegressor(max_depth=20)
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

# 232, without visits and appt 140
##    model = Ridge(alpha=5, solver="cholesky")
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # 140
##    model = Lasso()
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

##    model = ElasticNet()
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

##    model = KNeighborsRegressor()
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

      # 100 not good
##    reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)
##    y_pred=reg.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # RandomForestRegressor 230 result<=15 220, without visits and appts 166
##    model = RandomForestRegressor(n_estimators = 600, random_state = 0)
##    model.fit(X_train, y_train)
##    y_pred = model.predict(X_test)
##    mape = MAPE(y_test,y_pred)

    # Create the parameter grid based on the results of random search 
##    param_grid = {
##        'bootstrap': [True],
##        'max_depth': [80, 90, 100, 110],
##        'max_features': [2, 3],
##        'min_samples_leaf': [3, 4, 5],
##        'min_samples_split': [8, 10, 12],
##        'n_estimators': [100, 200, 300, 1000]
##    }
##    # Create a based model
##    rf = RandomForestRegressor()
##    # Instantiate the grid search model
##    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
##                              cv = 3, n_jobs = -1, verbose = 2)
##
##    # Fit the grid search to the data
##    grid_search.fit(X_train, y_train)
##    print(grid_search.best_params_)


## result.csv
##    for test in X_test:
##        month.append(test[0])
##        year.append(test[1])
##        id.append(pracId)
##
##    for pred in y_pred:
##        pred_prod.append(round(pred,2))

    if mape <= 15:
        countVal=countVal+1
                    
    print("PracticeId: ", pracId ," MAPE: ", round(mape,2))
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

 
print("Practices under 15% Error: ",countVal)

##  dictionary of lists 
##dict = {'id': id, 'month': month, 'year': year, 'production' : pred_prod}
##dp = pd.DataFrame(dict)
### saving the dataframe
##dp.to_csv('dataset/result.csv', index=False)


