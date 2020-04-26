# Des: Analysis of tweets extracted from 65 twitter media pages (150k+ tweets).
#      Similarily, goal is to predict stock price direction
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *
from tabulate import tabulate
import seaborn as sn

# Source files (functions):
import functions_nlp as fns

##########################################################################
# A. EXTRACT: Read in cleaned dataset
##########################################################################
df_features = pd.read_csv(r".\media_data_cleaned.csv")

##########################################
# 2. Split data: Logistic regression and for Linear Regression
##########################################

df_features = df_features.dropna()
df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()
df_features_ind = df_features[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB', 'FAV_COUNT_DAY',
       'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM', 'FOLLOWERS']]

# -- Logistic Regression + RF?
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(df_features_ind, df_features['direction'], test_size=0.2, random_state=0)

##########################################
# Find Best Logistic Regression from All variables
##########################################

def logistic_regression(Xtrain, Xtest, ytrain, ytest, list_of_features):
    logit_model = LogisticRegression()
    logit_model.fit(Xtrain[list_of_features], ytrain)
    pred = logit_model.predict(Xtest[list_of_features])  # predcition
    accuracy = logit_model.score(Xtest[list_of_features], ytest) # Return the mean accuracy on the given test data and labels.
    prob = logit_model.predict_proba(Xtest[list_of_features]) #	Probability estimates.
    intercept_log = logit_model.intercept_
    coefs = [i for i in logit_model.coef_]
    return [pred, accuracy, prob, intercept_log, coefs, list_of_features]

all_fts = ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB', 'FAV_COUNT_DAY',
       'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM', 'FOLLOWERS']

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

logit_model_b = LogisticRegression()
logit_model_b.fit(data_train_log[best_sc1[1]], price_train_log)
pred = logit_model_b.predict(data_test_log[best_sc1[1]])  # predcition
accuracy = logit_model_b.score(data_test_log[best_sc1[1]], price_test_log) # Return the mean accuracy on the given test data and labels.

print("# -- Test Results - Logistic Regression: Best Models -- #")
print("Number of possible models: ", len(combos))
print("Highest Mean Accuracy: ", best_sc1[0])
print("Best model from ALL variables: ")
print(df1)
print(classification_report(price_test_log, pred))
print("Confusion matrix: ")
print(confusion_matrix(price_test_log, pred))

print("##########################################################")
print("##########################################################")
plt.figure()

sn.heatmap(confusion_matrix(price_test_log, pred), annot=True)
##########################################
# -- Find Best Logistic Regression from SENTIMENT variables (and all variables)
##########################################

sent_fts = ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']

logit_model2 = LogisticRegression()
logit_model2.fit(data_train_log[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']], price_train_log)
predS = logit_model2.predict(data_test_log[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']])  # predcition
accuracyS = logit_model2.score(data_test_log[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']],price_test_log)  # Return t

combos1 = []
for L in range(0, len(sent_fts)+1):
    for subset in itertools.combinations(sent_fts, L):
        combos1.append(list(subset))

acc1 = []
for i in range(1, len(combos1)):
    log = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, combos1[i])
    acc1.append([log[1], combos1[i], log[4]])

best_sc = sorted(acc1)[-1]
df = pd.DataFrame()
df['coef'] = best_sc[2][0]
df['fts'] = best_sc[1]

logit_modelbest = LogisticRegression()
logit_modelbest.fit(data_train_log[best_sc[1]], price_train_log)
predS = logit_modelbest.predict(data_test_log[best_sc[1]])  # predcition
accuracyS = logit_modelbest.score(data_test_log[best_sc[1]],price_test_log)  # Return t

confusion_matrix(price_test_log, predS)


plt.figure()
plt.plot(sorted([i[0] for i in acc1]))
plt.xlabel("Set of Combinations of Sentiment features")
plt.ylabel("Mean Accuracy")
plt.title("Sentiment Logistic Regression Models - Mean Accuracy")

print("# -- Test Results - Logistic Regression: Best Models -- #")
print("Highest Mean Accuracy: ", best_sc[0])
print("Best model from SENTIMENT variables: ")
print(df)
print(classification_report(price_test_log, predS))
print("Confusion matrix: ")
print(confusion_matrix(price_test_log, predS))
print("##########################################################")
print("##########################################################")
plt.figure()
sn.heatmap(confusion_matrix(price_test_log, predS), annot=True)

print("# -- Grid Search Cross Validation -- #")

from sklearn.model_selection import GridSearchCV
logit_grid = LogisticRegression()

param_grid = {'C':  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator = logit_grid, param_grid = param_grid, cv = 10, n_jobs = -1)

grid_search.fit(df_features[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']], df_features['direction'])
opt_param = grid_search.best_params_


logit_modelbest = LogisticRegression(C=1)
logit_modelbest.fit(data_train_log[best_sc[1]], price_train_log)
predS = logit_modelbest.predict(data_test_log[best_sc[1]])  # predcition
accuracyS = logit_modelbest.score(data_test_log[best_sc[1]],price_test_log)  # Return t



