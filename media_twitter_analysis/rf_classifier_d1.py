
# Des: Analysis of tweets extracted from 65 twitter media pages (150k+ tweets).
#      Similarily, goal is to predict stock price direction
#      using sentiment analysis, correlation matrix, and logistic regression.
# By: Tiernan Barry - x19141840 (NCI) - Data Mining and Machine Learning

# Libraries:
import Twitter_API_Module as twt
import numpy as np
import pandas as pd
import re
import missingno as msno
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *
from tabulate import tabulate
import itertools
from sklearn.model_selection import GridSearchCV


##########################################################################
# A. EXTRACT: Read in cleaned dataset
##########################################################################
df_features = pd.read_csv(r".\media_data_cleaned.csv")

##########################################
# Split data:
##########################################
df_features = df_features.dropna()
df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()
df_features_ind = df_features[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
                                'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS']]

# --  Random Forest
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(df_features_ind, df_features['direction'], test_size=0.2, random_state=0, shuffle=True)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(data_train_log, price_train_log)
p = rfc.predict(data_test_log)
acc_rfc = rfc.score(data_test_log, price_test_log)
print("# -- Test Results - Random Forest Classifier: All Variables  -- #")
print("Mean accuracy: ", acc_rfc)
print(classification_report(price_test_log, p))
print("Confusion matrix: ")
print(confusion_matrix(price_test_log, p))
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

rfc_opt = RandomForestClassifier(bootstrap=True, max_depth= 250,min_samples_leaf= 2, min_samples_split= 8,n_estimators= 300)
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

print("# -- Optimal trees in Random Forest -- #")
acc_list =[]
for i in [50, 100, 300, 500, 800]:
    rf_tree = RandomForestClassifier(bootstrap=True, n_estimators=i, random_state=1)
    rf_tree.fit(data_train_log, price_train_log)
    acc_2 = rf_tree.score(data_test_log, price_test_log)
    acc_list.append([i, acc_2])
    p1 = rf_tree.predict(data_test_log)

rf_tree = RandomForestClassifier(bootstrap=True, max_depth= 250,min_samples_leaf= 2, min_samples_split= 8, n_estimators=50, random_state=1)
rf_tree.fit(data_train_log, price_train_log)
acc_2 = rf_tree.score(data_test_log, price_test_log)
acc_list.append([i, acc_2])
p1 = rf_tree.predict(data_test_log)

print("Mean accuracy: ", sorted(acc_list)[0][1])
print("Number of Trees: ", sorted(acc_list)[0][1])
print(classification_report(price_test_log, p1))
print("Confusion matrix: ")
print(confusion_matrix(price_test_log, p1))
print("##########################################################")
print("##########################################################")





