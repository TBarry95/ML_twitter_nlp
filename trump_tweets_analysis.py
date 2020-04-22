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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# Source files (functions):
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

##########################################################################
# EXTRACT:
##########################################################################
all_data = pd.read_csv(r".\trump_data_cleaned.csv")

##########################################################################
# ANALYSIS:
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 2. Split data:
# 3. PCA
# 4. Logistic Regression: Predict Stock price Direction
# -- Find Best model from All variables
# 5. Linear Regression: Predict Stock prices
# -- Find Best model from All variables
# 6. Random Forest Regression: Predict Stock prices
# -- Find Best model from All variables
##########################################################################

##########################################
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = all_data[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                       'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                       'DIRECTION2', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT',
                       'SP_CLOSE', 'SP_DIRECTION']].corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

##########################################
# 2. Split data:
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
# 3. PCA
##########################################
# --  Random Forest
fts_for_pca_trn = data_train_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                   'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']]
fts_for_pca_test = data_test_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                   'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']]

# -- Initialise PCA class
pca = PCA()
data_reduced_train_rf = pca.fit_transform(scale(fts_for_pca_trn))
data_reduced_test_rd = pca.transform(scale(fts_for_pca_test))

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')

print("# -- Test Results - PCA: Variance Explained per PC -- #")
print(variance_explained_2)
print("##########################################################")

# -- Logistic Regression
fts_for_pca_log = data_train_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                   'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']]
fts_for_pca_test_log = data_test_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                   'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']]

# -- Initialise PCA class
pca2 = PCA()
data_reduced_train_log = pca2.fit_transform(scale(fts_for_pca_log))
data_reduced_test_log = pca2.transform(scale(fts_for_pca_test_log))

##########################################
# 4.  Logistic Regression: Predict Stock price Direction
##########################################

##########################################
#  PCR: Logistic Regression: Predict Stock price Direction
##########################################

acc = []
for i in range(1, 11):
    logit_model = LogisticRegression()
    logit_model.fit(data_reduced_train_log[:, :i], price_train_log)
    pred = logit_model.predict(data_reduced_test_log[:, :i])  # predcition
    accuracy = logit_model.score(data_reduced_test_log[:, :i], price_test_log)  # Return the mean accuracy on the given test data and labels.
    intercept_log = logit_model.intercept_
    coefs = logit_model.coef_
    acc.append([i, accuracy,intercept_log,coefs  ])

acc_list = [i[1] for i in acc]
plt.figure()
plt.plot(acc_list)
plt.xlabel("Number of Principal Components")
plt.ylabel("Mean Accuracy")
plt.title("All PC Logistic Regression Models - Mean Accuracy")

print("# -- Test Results: Optimal PC Logistic Regression -- #")
print("Number of Principal Components: ", 1)
print("Mean Accuracy: ", sorted(acc_list)[-1])
print("##########################################################")
print("##########################################################")

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

##########################################
# 4. Random forest Classifier:
##########################################

rfc = RandomForestClassifier( n_estimators=1000, random_state=0)
rfc.fit(data_reduced_train_log, price_train_log)
p = rfc.predict(data_reduced_test_log)
acc_rfc = rfc.score(data_reduced_test_log, price_test_log)

print("# -- Test Results - Random forest Classifier: ")
print("Mean Accuracy: ", acc_rfc)
print("##########################################################")
print("##########################################################")
##########################################
# 4. Linear Regression: Predict Stock price
##########################################
# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
#                    'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
#                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT'

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
rf.fit(data_train_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']], price_train_rf)
rf_pred = rf.predict(data_test_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']])
rf.score(data_test_rf[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB',
                   'MEAN_SENT3_NB_PCT', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2', 'DIRECTION1',
                    'DIRECTION2', 'DIRECTION3', 'TWEET_COUNT']], price_test_rf)


def linear_regression(data, list_of_features, pred_days):

    # -- Set dependent variable and drop from feature set
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    validation = data[len(data)-pred_days:]
    data_TT = data[:len(data)-pred_days]
    dep_var_lm = data_TT['SP_CLOSE']

    # -- train and test:
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_TT, dep_var_lm, test_size=0.3, random_state=0)
    date_train = X_train2['DATE']
    date_test = X_test2['DATE']
    X_train2 = X_train2[list_of_features]
    X_test2 = X_test2[list_of_features]
    linear_model = LinearRegression()
    linear_model.fit(X_train2, y_train2)
    pred = linear_model.predict(X_test2)
    # -- compare prices:
    df_compare = pd.DataFrame()
    df_compare['DATE'] = date_test
    df_compare['PREDICTED_PX'] = pred
    df_compare['ACTUAL_PX'] = y_test2

    # -- validate:
    validation_fts = validation[list_of_features]
    validation_px = validation['SP_CLOSE']
    val_pred = linear_model.predict(validation_fts)
    # -- compare prices:
    df_compare_val = pd.DataFrame()
    df_compare_val['DATE'] = validation['DATE']
    df_compare_val['PREDICTED_PX'] = val_pred
    df_compare_val['ACTUAL_PX'] = validation_px
    return [df_compare, df_compare_val]

linear_pred1 = linear_regression(all_data, ['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'TWEET_COUNT', 'RT_COUNT' ], 40)
linear_pred2 = linear_regression(all_data, ['MEAN_DAILY_SENT2', 'TWEET_COUNT', 'RT_COUNT' ], 100)

plt.figure()
plt1, = plt.plot([i for i in range(0, len(linear_pred1[1]['DATE']))], linear_pred1[1]['PREDICTED_PX'])
plt2, = plt.plot([i for i in range(0, len(linear_pred1[1]['DATE']))], linear_pred1[1]['ACTUAL_PX'])
plt.title("Actual vs Predicted S&P500 Price")
plt.xlabel("Days up until April 12th")
plt.ylabel("Prices (USD)")
plt.legend([plt1, plt2], ["Predicted Price", "Actual Price"])

##########################################
# 5. Random Forest Regression: Predict Stock price
##########################################
# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2',
#                            'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT', RT_COUNT

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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









