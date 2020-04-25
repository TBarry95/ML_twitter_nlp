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
df_features = pd.read_csv(r".\media_data_cleaned.csv",)


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
# 3. PCA:
##########################################

# 1. Logistic regression:
# -- Initialise PCA class
pca = PCA()
data_reduced_train = pca.fit_transform(scale(data_train_log))
data_reduced_test = pca.transform(scale(data_test_log))

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')

df_pcs_train_log = pd.DataFrame(data=data_reduced_train, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])
df_pcs_test_log = pd.DataFrame(data=data_reduced_test, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])

##########################################################
# Logistic Regression PCA
##########################################################

logit_model = LogisticRegression()
logit_model.fit(df_pcs_train_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_train_log)
pred = logit_model.predict(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])  # predcition
accuracy = logit_model.score(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],price_test_log) # Return the mean accuracy on the given test data and labels.
prob = logit_model.predict_proba(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]) #	Probability estimates.

# -- Find Metrics and Visualise:
print("# -- Test Results - PCA: 5 PC Logistic Regression -- #")
print("Mean accuracy: ", accuracy)
# -- Print Equation:
intercept_log = logit_model.intercept_
coefs = logit_model.coef_
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
print("Logit 5 PC's = ", intercept_log[0], '+ (', pc[0],round(coefs[0][0],3), ') + (' ,pc[1],round(coefs[0][1],3), ') + (', pc[2],round(coefs[0][2],3),
      ') + (' ,pc[3],round(coefs[0][3],3), ') + (', pc[4],round(coefs[0][4],3), ')')
print("##########################################################")
print("##########################################################")
