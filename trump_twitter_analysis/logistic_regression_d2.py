# Des: Analysis script of Donald Trumps tweets in order to predict stock price direction
#      using sentiment analysis, correlation matrix, and logistic regression.
# By: Tiernan Barry - x19141840 (NCI) - Data Mining and Machine Learning

# Libraries:
import Twitter_API_Module as twt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

##########################################################################
# EXTRACT:
##########################################################################
all_data = pd.read_csv(r".\trump_data_cleaned.csv")

##########################################
# Split data:
##########################################

ind_vars = all_data[[ 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                   'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT',  'FAV_COUNT', 'RT_COUNT']]

dep_var1 = all_data['SP_CLOSE']
dep_var2 = all_data['SP_DIRECTION']

# --  Random Forest
data_train_rf, data_test_rf, price_train_rf, price_test_rf = train_test_split(ind_vars,dep_var1 , test_size=0.2, random_state=0, shuffle=True)
# -- Logistic Regression
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(ind_vars, dep_var2, test_size=0.2, random_state=0, shuffle=True)

##########################################
#  Logistic Regression: Sentiment / other values:
##########################################

# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
#                    'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
#                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT'
# -- From Sentiment results only:

def logistic_regression(Xtrain, Xtest, ytrain, ytest, list_of_features):
    logit_model = LogisticRegression()
    logit_model.fit(Xtrain[list_of_features], ytrain)
    pred = logit_model.predict(Xtest[list_of_features])  # predcition
    accuracy = logit_model.score(Xtest[list_of_features], ytest) # Return the mean accuracy on the given test data and labels.
    prob = logit_model.predict_proba(Xtest[list_of_features]) #	Probability estimates.
    intercept_log = logit_model.intercept_
    coefs = logit_model.coef_
    return [pred, accuracy, prob, intercept_log, coefs, list_of_features]

out_all_sent = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB'])
out_1 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_DAILY_SENT1'])
out_2 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_DAILY_SENT2'])
out_3 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_DAILY_SENT3_NB'])
print("# -- Test Results - Logistic Regression: Sentiment Features -- #")
print("Features: ", ['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB'])
print("Mean accuracy: ", out_all_sent[1])
print("Features: ", ['MEAN_DAILY_SENT1'])
print("Mean accuracy: ", out_1[1])
print("Features: ", ['MEAN_DAILY_SENT2'])
print("Mean accuracy: ", out_2[1])
print("Features: ", ['MEAN_DAILY_SENT3_NB'])
print("Mean accuracy: ", out_3[1])
print("##########################################################")
print("##########################################################")
