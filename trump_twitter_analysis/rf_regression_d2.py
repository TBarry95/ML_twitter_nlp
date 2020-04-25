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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

##########################################################################
# EXTRACT:
##########################################################################
all_data = pd.read_csv(r".\trump_data_cleaned.csv")

##########################################
# 2. Split data:
##########################################

ind_vars = all_data[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT',  'FAV_COUNT', 'RT_COUNT']]

dep_var1 = all_data['SP_CLOSE']
dep_var2 = all_data['SP_DIRECTION']

# --  Random Forest
data_train_rf, data_test_rf, price_train_rf, price_test_rf = train_test_split(ind_vars,dep_var1 , test_size=0.2, random_state=0, shuffle=True)
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(ind_vars, dep_var2, test_size=0.2, random_state=0, shuffle=True)

##########################################
# 4. RF Regression: Predict Stock price
##########################################
# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
#                    'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
#                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT'

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
rf.fit(data_train_rf[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT']], price_train_rf)
rf_pred = rf.predict(data_test_rf[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT']])
rf.score(data_test_rf[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB', 'TWEET_COUNT']], price_test_rf)

scaler = StandardScaler()

# -- Set dependent variable and drop from feature set
all_data = all_data.replace([np.inf, -np.inf], np.nan)
all_data = all_data.dropna()
validation = all_data[len(all_data) - 40:]
data_TT = all_data[:len(all_data) - 40]
dep_var_lm = data_TT['SP_CLOSE']

# -- train and test:
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_TT, dep_var_lm, test_size=0.3, random_state=0)
date_train = X_train2['DATE']
date_test = X_test2['DATE']
del X_test2['DATE']
del X_train2['DATE']
del X_test2['DATE_TIME']
del X_train2['DATE_TIME']

X_train2 = X_train2[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'TWEET_COUNT', 'RT_COUNT']]

X_test2 = X_test2[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'TWEET_COUNT','RT_COUNT']]

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)

# Train the model on training data
rf.fit(X_train2, y_train2)

rf.predict(X_test2)
rf.score(X_test2, y_test2)

'''
df = pd.DataFrame()
df['DATE'] = validation['DATE']
df['PREDICTED_PX'] = rf.predict(validation[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'TWEET_COUNT', 'RT_COUNT']])
df['ACTUAL_PX'] = validation['SP_CLOSE']

plt.figure()
plt1, = plt.plot([i for i in range(0, len(df['DATE']))],df['PREDICTED_PX'])
plt2, = plt.plot([i for i in range(0, len(df['DATE']))], df['ACTUAL_PX'] )
plt.title("Actual vs Predicted S&P500 Price")
plt.xlabel("Days up until April 12th")
plt.ylabel("Prices (USD)")
plt.legend([plt1, plt2], ["Predicted Price", "Actual Price"])
'''




