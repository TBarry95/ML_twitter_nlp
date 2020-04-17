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
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# Source files (functions):
import functions_nlp as fns

##########################################################################
# Extract:
##########################################################################

# -- Read in tweets as sourced from get_datasets.py
df_all_tweets = pd.read_csv(r"C:\Users\btier\Documents\twitter_mass_media_data_2.csv")
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")
gspc_df['pct_change'] = gspc_df['Close'].pct_change()
gspc_df['direction'] = [1 if i > 0 else 0 for i in gspc_df['pct_change']]

##########################################################################
# Transform:
##########################################################################

# -- Format date:
df_all_tweets['DATE_TIME'] = [str(i)[0:10] for i in df_all_tweets['DATE_TIME']]

# -- Make new column for processed name:
df_all_tweets['PROCESSED_TEXT'] = df_all_tweets['FULL_TEXT'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# -- Check for formatting:
word_cloud = fns.get_wordcloud(df_all_tweets, r"C:\Users\btier\Documents\news_word_cloud.png")

# -- bag of words - stop words already removed:
top_words = fns.get_top_words(df_all_tweets)

##########################################################################
# Analysis:
# 1. Get Sentiment: Lexicon-based polarity
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 3. Logistic Regression: Predict Stock price Direction
# 4. Linear Regression: Predict Stock price
# 5. Random Forest Regression: Predict Stock price
##########################################################################

##########################################
# 1. Get Sentiment: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1,0,1):
df_all_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in df_all_tweets["PROCESSED_TEXT"]])
df_all_tweets = fns.get_sentiment_pa(df_all_tweets)

# -- Get feature set: Aggregate tweets by date:
df_features = pd.DataFrame()
df_features['MEAN_SENT1'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].mean()
df_features['MEAN_SENT2'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_PA'].mean()
df_features['MEAN_SENT1_PCT'] = df_features['MEAN_SENT1'].pct_change()
df_features['MEAN_SENT2_PCT'] = df_features['MEAN_SENT2'].pct_change()
df_features['FAV_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['FAV_COUNT'].sum()
df_features['RT_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['RT_COUNT'].sum()
df_features['TWEET_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].count()
df_features['LEN_TWEET_SUM'] = df_all_tweets.groupby('DATE_TIME')['LEN_TWEET'].sum()
df_features['FOLLOWERS'] = df_all_tweets.groupby('DATE_TIME')['FOLLOWERS'].sum()

# -- Join tweets to stock prices:
gspc_df_features = gspc_df[['Date', 'Close', 'pct_change', 'direction']]
df_features = pd.merge(df_features, gspc_df_features, how='left', left_on='DATE_TIME', right_on='Date')
msno.matrix(df_features, figsize= (50,30))
df_features = df_features.dropna()

##########################################
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = df_features[['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT',
       'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
       'FOLLOWERS', 'Close']].corr()

mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


##########################################
# 4. K-fold ross validation:
##########################################
df_features = df_features.dropna()

ind_vars = df_features[['Date', 'MEAN_SENT1', 'MEAN_SENT2', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM']]
dep_var = df_features['Close']



data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(df_features, dep_var, test_size=0.2, random_state=0)
del data_train['Date']

# -- Initialise PCA class
pca = PCA()

reduce = pca.fit_transform(data_train)



# -- Standardise/scale the training data such that each column's mean = 0
data_reduced_train = pca.fit_transform(scale(data_train))
print(pd.DataFrame(pca.components_.T).head())

# -- Find optimal number of components by applying k-fold Cross Validation
kfold_cv_10_2 = model_selection.KFold(n_splits=10, random_state=0, shuffle=True)

# -- Initialise LR model
lr_model_pca = LinearRegression()

# -- Use MSE as an indicator for closest fit:
mse_pca_2 = []

# -- Looping through X number of PC's, appends the MSE to above list. Will be used to find best model.
for i in np.arange(1, 50):
    # Multiple by -1 to negate the scoring method
    mse_result = -1 * model_selection.cross_val_score(lr_model_pca, data_reduced_train[:, :i], gspc_px_train.ravel(),
                                                      cv=kfold_cv_10_2,scoring='neg_mean_squared_error').mean()
    mse_pca_2.append(mse_result)

# -- Plot elbow graph of MSE
plt.figure()
plt.plot(mse_pca_2, '-v')
plt.xlabel('Principal Components in Linear Regression Model')
plt.ylabel('MSE - Mean Squared Error')
plt.title('Elbow Chart - PCA K-Fold Cross Validation (Training)')

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Linear Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')





##########################################
# 3. Logistic Regression: Predict Stock price Direction
##########################################
'''['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT',
       'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
       'FOLLOWERS',],'''

def logistic_regression(data, list_of_features):

    # -- Set dependent variable and drop from feature set
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    dep_var = data['direction']

    # -- All variables + clean:
    vars_for_logit = data[list_of_features]

    # -- Run Logistic Regression model 1:
    X_train1, X_test1, y_train1, y_test1 = train_test_split(vars_for_logit, dep_var, test_size=0.3, random_state=0)
    logit_model = LogisticRegression()
    logit_model.fit(X_train1, y_train1)
    pred = logit_model.predict(X_test1)  # predcition
    accuracy = logit_model.score(X_test1, y_test1) # Return the mean accuracy on the given test data and labels.
    prob = logit_model.predict_proba(X_test1) #	Probability estimates.

    return [pred, accuracy, prob]
log_out1 = logistic_regression(df_features, ['MEAN_SENT2_PCT'])
log_out2 = logistic_regression(df_features, ['MEAN_SENT1', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM', 'MEAN_SENT1_PCT'])

##########################################
# 4. Linear Regression: Predict Stock price
##########################################

def linear_regression(data, list_of_features, pred_days):

    # -- Set dependent variable and drop from feature set
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    validation = data[len(data)-pred_days:]
    data_TT = data[:len(data)-pred_days]
    dep_var_lm = data_TT['Close']

    # -- train and test:
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_TT, dep_var_lm, test_size=0.3, random_state=0)
    date_train = X_train2['Date']
    date_test = X_test2['Date']
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
    validation_px = validation['Close']
    val_pred = linear_model.predict(validation_fts)
    # -- compare prices:
    df_compare_val = pd.DataFrame()
    df_compare_val['DATE'] = validation['Date']
    df_compare_val['PREDICTED_PX'] = val_pred
    df_compare_val['ACTUAL_PX'] = validation_px
    return [df_compare, df_compare_val]

lin_out2 = linear_regression(df_features,  ['MEAN_SENT1', 'LEN_TWEET_SUM'],40)






lr_model = LinearRegression()

df_features = df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()

dep_var = df_features['Close']

# -- All variables + clean:
vars_for_logit = df_features[['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT',
                             'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                             'FOLLOWERS']]

# -- Run Logistic Regression model 1:
X_train1, X_test1, y_train1, y_test1 = train_test_split(vars_for_logit, dep_var, test_size=0.3, random_state=0)
lr_model.fit(X_train1, y_train1)
pred = lr_model.predict(X_test1)

lr_model.score(X_test1, y_test1)


