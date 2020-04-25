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
    coefs = logit_model.coef_
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
    acc.append([log[1], combos[i]])

# -- results of best
all_acc = [i[0] for i in acc]
all_fts = [i[1] for i in acc]
all_acc = sorted(all_acc)
highest_acc = all_acc[-1]

plt.figure()
plt.plot(all_acc)
plt.xlabel("Set of Combinations of all features")
plt.ylabel("Mean Accuracy")
plt.title("All Possible Logistic Regression Models - Mean Accuracy")

# -- get fts
df = pd.DataFrame()
df['all_mean'] = [i for i in all_acc]
df['fts'] = [i for i in all_fts]
df['len'] = [len(i) for i in df['fts']]
df_best = df[df['all_mean'] == highest_acc]

lowest = sorted(df_best['len'])
lowest = lowest[0]
df_best7 = df_best[df_best['len']<=lowest]

print("# -- Test Results - Logistic Regression: Best Models -- #")
print("Highest Mean Accuracy: ", highest_acc)
print("Number of Best Models", len(df_best7['all_mean'][df_best7['all_mean']==highest_acc]))
print("##########################################################")
print("##########################################################")
print("# -- Best Logistic Regression Models across all variables: -- #")
print("     MEAN ACCURACY           COMBINATION OF PREDICTORS")
print(tabulate(df_best7))
##########################################
# -- Find Best Logistic Regression from SENTIMENT variables (and all variables)
##########################################

sent_fts = ['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB']

combos1 = []
for L in range(0, len(sent_fts)+1):
    for subset in itertools.combinations(sent_fts, L):
        combos1.append(list(subset))

acc1 = []
for i in range(1, len(combos1)):
    log = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log, combos1[i])
    acc1.append([log[1], combos1[i]])

# -- results of best
all_acc1 = [i[0] for i in acc1]
all_fts1 = [i[1] for i in acc1]
all_acc1 = sorted(all_acc1)
highest_acc1 = all_acc1[-1]
lowest_acc1 = all_acc1[0]

plt.figure()
plt.plot(all_acc1)
plt.xlabel("Set of Combinations of Sentiment features")
plt.ylabel("Mean Accuracy")
plt.title("Sentiment Logistic Regression Models - Mean Accuracy")

# -- get fts
df1 = pd.DataFrame()
df1['all_mean'] = [i for i in all_acc1]
df1['fts'] = [i for i in all_fts1]
df1_best = df1[df1['all_mean'] == highest_acc1]

df1_best['len'] = [len(i) for i in df1_best['fts']]
lowest = sorted(df1_best['len'])
lowest = lowest[0]
df_best3 = df1_best[df1_best['len']==lowest]

print("# -- Test Results - Logistic Regression: Best Models -- #")
print("Highest Mean Accuracy: ", highest_acc1)
print("Number of Best Models", len(df_best3['all_mean'][df_best3['all_mean']==highest_acc1]))
print("##########################################################")
print("##########################################################")
print("# -- Best Logistic Regression Models across all variables: -- #")
print("     MEAN ACCURACY           COMBINATION OF PREDICTORS")
print(tabulate(df_best3))

