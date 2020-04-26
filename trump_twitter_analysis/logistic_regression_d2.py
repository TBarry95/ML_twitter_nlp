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

ind_vars = all_data[[ 'MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB', 'TWEET_COUNT',  'FAV_COUNT', 'RT_COUNT']]

dep_var1 = all_data['SP_CLOSE']
dep_var2 = all_data['SP_DIRECTION']

# --  Random Forest
data_train_rf, data_test_rf, price_train_rf, price_test_rf = train_test_split(ind_vars,dep_var1 , test_size=0.2, random_state=0, shuffle=True)
# -- Logistic Regression
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(ind_vars, dep_var2, test_size=0.2, random_state=0, shuffle=True)

##########################################
#  Logistic Regression: Sentiment / other values:
##########################################

# Possible Features: 'MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
#        'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
#        'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT'
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

out_all_sent = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB'])
out_1 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_SENT_LEX_CATG'])
out_2 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_SENT_LEX_POL'])
out_3 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, ['MEAN_SENT_NB'])
print("# -- Test Results - Logistic Regression: Sentiment Features -- #")
print("Features: ", ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB'])
print("Mean accuracy: ", out_all_sent[1])
print("Features: ", ['MEAN_SENT_LEX_CATG'])
print("Mean accuracy: ", out_1[1])
print("Features: ", ['MEAN_SENT_LEX_POL'])
print("Mean accuracy: ", out_2[1])
print("Features: ", ['MEAN_SENT_NB'])
print("Mean accuracy: ", out_3[1])
print("##########################################################")
print("##########################################################")

'''
all_fts = ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT']

combos = []
import itertools
for L in range(0, len(all_fts)+1):
    for subset in itertools.combinations(all_fts, L):
        combos.append(list(subset))

acc = []
for i in range(1, len(combos)):
    log = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, combos[i])
    acc.append([log[1], combos[i], log[4]])


best_sc1 = sorted(acc)[-1]
df1 = pd.DataFrame()
df1['coef'] = best_sc1[2][0]
df1['fts'] = best_sc1[1]

plt.figure()
plt.plot([i[0] for i in acc])
plt.xlabel("Set of Combinations of all features")
plt.ylabel("Mean Accuracy")
plt.title("All Possible Logistic Regression Models - Mean Accuracy")


'''



print("# -- Grid Search Cross Validation -- #")
from sklearn.model_selection import GridSearchCV
logit_grid = LogisticRegression()

param_grid = {'C':  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator = logit_grid, param_grid = param_grid, cv = 10, n_jobs = -1)

grid_search.fit(data_train_log, data_test_log)
opt_param = grid_search.best_params_


logit_modelbest = LogisticRegression(C=1)
logit_modelbest.fit(data_train_log[best_sc[1]], price_train_log)
predS = logit_modelbest.predict(data_test_log[best_sc[1]])  # predcition
accuracyS = logit_modelbest.score(data_test_log[best_sc[1]],price_test_log)  # Return t



