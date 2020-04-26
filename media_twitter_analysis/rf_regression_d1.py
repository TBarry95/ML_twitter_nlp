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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *
from tabulate import tabulate
import itertools

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
data_train_lin, data_test_lin, price_train_lin, price_test_lin = train_test_split(df_features_ind, df_features['Close'], test_size=0.2, random_state=0, shuffle=True)
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(df_features_ind, df_features['direction'], test_size=0.2, random_state=0, shuffle=True)

##########################################
# Random Forest Regression: Predict Stock price
##########################################

# -- 1. Predict Stock price
rf_model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_model.fit(data_train_lin, price_train_lin)
pred_rf  = rf_model.predict(data_test_lin)
acc_rf = rf_model.score(data_test_lin, price_test_lin)

print("# -- Test Results - Random Forest: All variables -- #")
print('Mean Squared Error:', mean_squared_error(price_test_lin, pred_rf))
print('Mean Absolute Error:', mean_absolute_error(price_test_lin, pred_rf))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(price_test_lin, pred_rf)))
print('R-Squared:', r2_score(price_test_lin, pred_rf))
print('Median Absolute Error:', median_absolute_error(price_test_lin, pred_rf))
print("##########################################################")
print("##########################################################")

plt.figure()
pl1, = plt.plot([i for i in range(0,len(pred_rf))],pred_rf)
pl2, = plt.plot([i for i in range(0,len(price_test_lin))],price_test_lin )
plt.legend([pl1, pl2], ["Random Forest Price", "Actual Price"])
plt.title("Random Forest vs Actual S&P500 price (shuffled results)")
plt.xlabel("Random sample of dates")
plt.ylabel("Price (USD)")

print("# -- Random Forest: Important Variables -- #")
feature_imp = pd.Series(rf_model.feature_importances_, index= data_test_lin.columns).sort_values(ascending=False)
print(feature_imp)
print("##########################################################")
print("##########################################################")
