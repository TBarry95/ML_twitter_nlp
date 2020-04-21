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
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import csv
import sys
nltk.download('stopwords')
nltk.download('punkt')
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

##########################################################################
# C. ANALYSIS:
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 2. Split data: Logistic regression and for Linear Regression
# 3. PCA
# 4. Logistic Regression: Predict Stock price Direction:
# -- PCA logistic
# -- Find Best model from All variables
# 5. Random Forest Regression: Predict Stock price
# -- Find Best model from All variables
##########################################################################

##########################################
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = df_features[['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
       'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
       'FOLLOWERS', 'Close']].corr()

mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print("# -- Results: Correlation Matrix: -- #")
print(corr_mx)
print("No significant correlation between SP500 close and any tweet sentiment metrics (although positive")
print("##########################################################")
print("##########################################################")

##########################################
# 2. Split data: Logistic regression and for Linear Regression
##########################################

df_features = df_features.dropna()
df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()
df_features_ind = df_features[['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS']]

'''# -- Extract validation subset: Keeping for last - never tested on
validation_data = df_features_ind[int(len(df_features_ind)*0.95):]
validation_px_rf  = df_features['Close'][int(len(df_features_ind)*0.95):]
validation_px_log  = df_features['direction'][int(len(df_features_ind)*0.95):]
# -- Test / Train split:
non_validation_data = df_features_ind[:int(len(df_features_ind)*0.95)]
non_validation_pxrf  = df_features['Close'][:int(len(df_features_ind)*0.95)]
non_validation_pxlog  = df_features['direction'][:int(len(df_features_ind)*0.95)]

# --  Random Forest
data_train_lin, data_test_lin, price_train_lin, price_test_lin = train_test_split(non_validation_data, non_validation_pxrf, test_size=0.3, random_state=0)
# -- Logistic Regression + RF?
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(non_validation_data, non_validation_pxlog, test_size=0.3, random_state=0)'''

# --  Random Forest
data_train_lin, data_test_lin, price_train_lin, price_test_lin = train_test_split(df_features_ind, df_features['Close'], test_size=0.2, random_state=0, shuffle=True)
# -- Logistic Regression + RF?
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(df_features_ind, df_features['direction'], test_size=0.2, random_state=0)

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

##########################################
# 4. Logistic Regression: Predict Stock price Direction
##########################################
# -- PCA logistic
# -- Find Best Logistic Regression from All variables
# -- Find Best Logistic Regression from sentiment variables
##########################################
# -- Get predicition using 5 PC's: PCA
##########################################

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

##########################################
# -- Find Best Logistic Regression from All variables
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

all_fts = ['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS']

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


##########################################
# -- Find Best Logistic Regression from SENTIMENT variables (and all variables)
##########################################

sent_fts = ['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT','MEAN_SENT3_NB']

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

##########################################
# 5. Random Forest Regression: Predict Stock price and Direction:
##########################################
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

##########################################
# -- Find Best Random Forest Regression All
##########################################

def random_forest_regression(Xtrain, Xtest, ytrain, ytest, list_of_features, num):
    rf = RandomForestRegressor(n_estimators = num, random_state = 0)
    rf.fit(Xtrain[list_of_features], ytrain)
    pred = rf.predict(Xtest[list_of_features])  # predcition
    accuracy = rf.score(Xtest[list_of_features], ytest) # Return the mean accuracy on the given test data and labels.
    return [pred, accuracy, list_of_features]

all_fts = ['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS']

combosrf = []
for L in range(0, len(all_fts)+1):
    for subset in itertools.combinations(all_fts, L):
        combosrf.append(list(subset))

accrf = []
for i in range(1, len(combosrf)):
    log = random_forest_regression(data_train_lin, data_test_lin, price_train_lin, price_test_lin, combosrf[i], 1000)
    accrf.append([log[1], combosrf[i]])

# -- results of best
all_acc_rf = [i[0] for i in accrf]
all_fts_rf = [i[1] for i in accrf]
all_acc_rf = sorted(all_acc_rf)
highest_acc_rf = all_acc_rf[-1]

plt.figure()
plt.plot(all_acc_rf)
plt.xlabel("Set of Combinations of all features")
plt.ylabel("Mean Accuracy")
plt.title("All Possible Random Forest Regression Models - R-squared")

# -- get fts
dfrf = pd.DataFrame()
dfrf['all_mean'] = [i for i in all_acc_rf]
dfrf['fts'] = [i for i in all_fts_rf]
dfrf['len'] = [len(i) for i in dfrf['fts']]
dfrf_best = dfrf[dfrf['all_mean'] == highest_acc_rf]

lowest = sorted(dfrf_best['len'])
lowest = lowest[0]
dfrf_best = dfrf_best[dfrf_best['len']<=lowest]

print("# -- Test Results - Random Forest Regression: Best Model -- #")
print("Highest R-squared: ", highest_acc_rf)
print("Number of Best Models", len(dfrf_best['all_mean'][dfrf_best['all_mean']==highest_acc_rf]))
print("##########################################################")
print("##########################################################")

print("# -- Best Random Forest Regression Model across all variables: -- #")
print("     MEAN ACCURACY           COMBINATION OF PREDICTORS")
print(tabulate(dfrf_best))

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
'''
print("# -- Random Forest: Contribution for Decision (egs - All variables) -- #")
predictions_egs = data_test_lin[0:2]
prediction, bias, contributions = ti.predict(rf_model, predictions_egs)
for i in range(len(predictions_egs)):
    print("Prediction", i)
    print( "Contribution by Top Feature:")
    for c, feature in sorted(zip(contributions[i], data_test_lin.columns))[0:2]:
        print(feature, round(c, 2))
    print( "-"*20)
print("##########################################################")
print("##########################################################")'''

# -- 2. Predict Direction: Better than logistic?
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(data_reduced_train_log, price_train_log)
p = rfc.predict(data_reduced_test_log)
acc_rfc = rfc.score(data_reduced_test_log, price_test_log)
print("# -- Test Results - Random Forest Classifier: All Variables  -- #")
print("Mean accuracy: ", acc_rf_pca_dir)
print("##########################################################")
print("##########################################################")


##########################################################################
# C. RESULTS:
##########################################################################
