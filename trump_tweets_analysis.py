# Des: Analysis script of Donald Trumps tweets in order to predict stock price direction
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import _logistic_loss
import seaborn as sns
from sklearn.metrics import *

# Source files (functions):
import functions_nlp as fns

##########################################################################
# Extract:
##########################################################################

# -- Read in Trump tweets:
trump_tweets = pd.read_csv(r"C:\Users\btier\PycharmProjects\DataMining_ML_virt\trump_tweets.csv")
trump_tweets.columns = ['SOURCE', 'FULL_TEXT', 'CREATED_AT', 'RETWEET_COUNT', 'FAV_COUNT', 'IS_RETWEET', 'ID_STR']
trump_tweets['CREATED_AT'] = [str(i)[0:10] for i in trump_tweets['CREATED_AT']]

# -- Read in SP500 price:
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")

# -- Get % change in daily prices, and boolean column of UP or DOWN
gspc_df['pct_change'] = [i*100 for i in gspc_df['Close'].pct_change()]
gspc_df['movement'] = [1 if i > 0 else 0 for i in gspc_df['pct_change']]

# -- Merge datasets by date:
trump_tweets['DATE_TIME'] = [str(i)[6:10] + '-' + str(i)[0:2] + str(i)[2:5] for i in trump_tweets['CREATED_AT']]
spx = pd.DataFrame({"DATE": gspc_df['Date'], "SP_CLOSE": gspc_df['Close'], "SP_PCT_CHANGE": gspc_df['pct_change'], "SP_DIRECTION": gspc_df['movement']})
trump_tweets = pd.merge(trump_tweets, spx, how='left', left_on='DATE_TIME', right_on='DATE')

del trump_tweets['CREATED_AT']
del trump_tweets['DATE']

##########################################################################
# Transform:
##########################################################################

# -- Deal with NA values: Back fill followed by forward fill
msno.matrix(trump_tweets, figsize= (50,30))
trump_tweets = trump_tweets.fillna(method='bfill')
trump_tweets = trump_tweets.fillna(method='ffill')
msno.matrix(trump_tweets, figsize= (50,30))

# -- Make new column for processed name:
trump_tweets['PROCESSED_TEXT'] = trump_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
trump_tweets.to_csv(r"C:\Users\btier\Documents\trump_processed.csv", index=False)

# -- Check for formatting using word cloud:
word_cloud = fns.get_wordcloud(trump_tweets, r"C:\Users\btier\Documents\trump_word_cloud.png")

##########################################################################
# Analysis:
# 1. Get Sentiment: Lexicon-based polarity
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 3. Logistic Regression: Predict Stock price Direction
# 3. Linear Regression: Predict Stock prices
##########################################################################

##########################################
# 1. Get Sentiment: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1,0,1):
trump_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in trump_tweets["PROCESSED_TEXT"]])
trump_tweets = fns.get_sentiment_pa(trump_tweets)

# -- Plot:
trump_tweets["SENTIMENT_1"].plot(kind='hist', legend=True)
trump_tweets["SENTIMENT_PA"].plot(kind='hist', legend=True)

# -- Get Average Sentiment for each date
df_feature_set = pd.DataFrame()
sent_data_1 = trump_tweets.groupby('DATE_TIME')['SENTIMENT_1'].mean()
sent_data_2 = trump_tweets.groupby('DATE_TIME')['SENTIMENT_PA'].mean()
count_tweets_day = trump_tweets.groupby('DATE_TIME')['SENTIMENT_1'].count()
rt_per_day = trump_tweets.groupby('DATE_TIME')['RETWEET_COUNT'].sum()
fav_per_day = trump_tweets.groupby('DATE_TIME')['FAV_COUNT'].sum()
df_feature_set['DATE'] = [i for i in sent_data_1.index]
df_feature_set['MEAN_DAILY_SENT1'] = [i for i in sent_data_1]
df_feature_set['MEAN_DAILY_SENT2'] = [i for i in sent_data_2]
df_feature_set['PCT_CHG_SENT1'] = df_feature_set['MEAN_DAILY_SENT1'].pct_change()
df_feature_set['PCT_CHG_SENT2'] = df_feature_set['MEAN_DAILY_SENT2'].pct_change()
df_feature_set['DIRECTION1'] = [1 if i > 0 else 0 for i in df_feature_set['PCT_CHG_SENT1']]
df_feature_set['DIRECTION2'] = [1 if i > 0 else 0 for i in df_feature_set['PCT_CHG_SENT2']]
df_feature_set['TWEET_COUNT'] = [i for i in count_tweets_day]
df_feature_set['FAV_COUNT'] = [i for i in fav_per_day]
df_feature_set['RT_COUNT'] = [i for i in rt_per_day]
df_feature_set = df_feature_set[len(df_feature_set)-1300:]

# -- Join prices
all_data = pd.merge(df_feature_set, trump_tweets[['DATE_TIME', 'SP_CLOSE', 'SP_DIRECTION']], how='left', left_on='DATE', right_on='DATE_TIME')
all_data = all_data.drop_duplicates()

# -- Clean
all_data = all_data.dropna()

##########################################
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = all_data[['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2','PCT_CHG_SENT1','PCT_CHG_SENT2',
                    'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT', 'FAV_COUNT', 'RT_COUNT', 'SP_CLOSE']].corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

##########################################
# 3. Logistic Regression: Predict Stock price Direction
##########################################

# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2',
#                            'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT', RT_COUNT

def logistic_regression(data, list_of_features):

    # -- Set dependent variable and drop from feature set
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    dep_var = data['SP_DIRECTION']

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
log_data1 = logistic_regression(all_data, ['MEAN_DAILY_SENT1'])
log_data2 = logistic_regression(all_data, ['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2',
                                             'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT', 'RT_COUNT'])

##########################################
# 4. Linear Regression: Predict Stock price
##########################################
# Possible Features: 'MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2',
#                            'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT', RT_COUNT
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
# Import the model we are using
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

'''X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.fit_transform(X_test2)'''

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)

# Train the model on training data
rf.fit(X_train2, y_train2)

rf.predict(X_test2)
rf.score(X_test2, y_test2)

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











