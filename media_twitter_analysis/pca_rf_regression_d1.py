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

# 2. RF regression:
# -- Initialise PCA class
pca1 = PCA()
data_reduced_train1 = pca1.fit_transform(scale(data_train_lin))
data_reduced_test1 = pca1.transform(scale(data_test_lin))

df_pcs_train_lin = pd.DataFrame(data=data_reduced_train1, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])
df_pcs_test_lin = pd.DataFrame(data=data_reduced_test1, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])

print("# -- Test Results - PCA: Variance Explained per PC -- #")
print(variance_explained_2)
print("##########################################################")
print("##########################################################")



# -- PCA Random Forest Regression
# -- Find Best Random Forest Regression All

##########################################
# -- Get predicition using 5 PC's: PCA
##########################################

from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti

# -- Using PCA:
rf_model_pc = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_model_pc.fit(df_pcs_train_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_train_lin)
pred_rf_pca  = rf_model_pc.predict(df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])
acc_rf_pca = rf_model_pc.score(df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_test_lin)
print("# -- Test Results - Random Forest Regression: PCA 5 variables -- #")
print('Mean Squared Error:', mean_squared_error(price_test_lin, pred_rf_pca))
print('Mean Absolute Error:', mean_absolute_error(price_test_lin, pred_rf_pca))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(price_test_lin, pred_rf_pca)))
print('R-Squared:', r2_score(price_test_lin, pred_rf_pca))
print('Median Absolute Error:', median_absolute_error(price_test_lin, pred_rf_pca))
print("##########################################################")
print("##########################################################")

print("# -- Random Forest: Contribution for Decision (30 egs - PCA) -- #")
predictions_egs_pca = df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']][0:2]
prediction_pc, bias, contributions_pc = ti.predict(rf_model_pc, predictions_egs_pca)
for i in range(len(predictions_egs_pca)):
    print("Prediction", i)
    print( "Contribution by Top Feature:")
    for c, feature in sorted(zip(contributions_pc[i], df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].columns))[0:2]:
        print(feature, round(c, 2))
    print( "-"*20)
print("##########################################################")
print("##########################################################")
