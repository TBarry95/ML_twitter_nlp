# Des: Analysis script of Donald Trumps tweets in order to predict stock price direction
#      using sentiment analysis, correlation matrix, and logistic regression.
# By: Tiernan Barry - x19141840 (NCI) - Data Mining and Machine Learning

# Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

##########################################################################
# EXTRACT:
##########################################################################
all_data = pd.read_csv(r".\trump_data_cleaned.csv")
all_data.dropna()
##########################################
# 2. Split data:
##########################################

ind_vars = all_data[[ 'MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT',  'RT_COUNT']]

dep_var1 = all_data['SP_CLOSE']
dep_var2 = all_data['SP_DIRECTION']

# --  Random Forest
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(ind_vars, dep_var2, test_size=0.2, random_state=0, shuffle=True)

##########################################
# 4. Random forest Classifier:
##########################################

rfc = RandomForestClassifier( n_estimators=1000, random_state=0)
rfc.fit(data_train_log, price_train_log)
p = rfc.predict(data_test_log)
acc_rfc = rfc.score(data_test_log, price_test_log)

print("# -- Test Results - Random forest Classifier: ")
print("Mean Accuracy: ", acc_rfc)
print("##########################################################")
print("##########################################################")


print("# -- Grid Search Corss validation -- #")

param_grid = {
    'bootstrap': [True],
    'max_depth': [30, 110,  250],
    'min_samples_leaf': [2, 4, 10],
    'min_samples_split': [6, 8, 12],
    'n_estimators': [300, 1000, 2000]
}

rfc = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1)
grid_search.fit(data_train_log, price_train_log)
opt_param = grid_search.best_params_

rfc_opt = RandomForestClassifier(bootstrap=True, max_depth= 250,min_samples_leaf= 2, min_samples_split= 6,n_estimators= 2000)
rfc_opt.fit(data_train_log, price_train_log)
acc_2 = rfc_opt.score(data_test_log, price_test_log)
p1  = rfc_opt.predict(data_test_log)

print("# -- Test Results - Random Forest: Grid search - Optimal Model -- #")
print(opt_param)
print("Mean accuracy: ", acc_2)
print(classification_report(price_test_log, p1))
print("Confusion matrix: ")
print(confusion_matrix(price_test_log, p1))
print("##########################################################")
print("##########################################################")



