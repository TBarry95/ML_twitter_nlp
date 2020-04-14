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
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import _logistic_loss
import seaborn as sns

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
#msno.matrix(trump_tweets, figsize= (50,30))
trump_tweets = trump_tweets.fillna(method='bfill')
trump_tweets = trump_tweets.fillna(method='ffill')
#msno.matrix(trump_tweets, figsize= (50,30))

# -- Make new column for processed name:
trump_tweets['PROCESSED_TEXT'] = trump_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
trump_tweets.to_csv(r"C:\Users\btier\Documents\trump_processed.csv", index=False)

# -- Check for formatting using word cloud:
word_cloud = fns.get_wordcloud(trump_tweets, r"C:\Users\btier\Documents\trump_word_cloud.png")
'''
# -- Plot top words - removed stop words:
top_words = fns.get_top_words(trump_tweets)
df_top_words = pd.DataFrame({"WORD": top_words[0], "COUNT": top_words[1]})
plt.figure()
plt.bar(df_top_words["WORD"][0:10], df_top_words["COUNT"][0:10])
plt.xlabel('Words', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title("Top 10 Words", fontsize=20)'''

##########################################################################
# Analysis:
# 1. Get Sentiment: Lexicon-based polarity
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 3. Logistic Regression: Predict Stock price Direction
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
#                            'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT'

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
                                             'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT'])


'''
import itertools
combos = []
for i in range(1, len(all)+1):
    for subset in itertools.combinations(all, i):
        combos.append(subset)


def get_all_logistic_regression(data):
    all = ['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'PCT_CHG_SENT1', 'PCT_CHG_SENT2',
                              'DIRECTION1', 'DIRECTION2', 'TWEET_COUNT']
    combos = []
    for i in range(1, len(all) + 1):
        for subset in itertools.combinations(all, i):
            combos.append(subset)

    df = pd.DataFrame()
    df['ACCURACY'] = []
    for i in range(0,len(combos)):
        feature = []
        for ii in combos[i]:
            feature.append(ii)

        # -- Set dependent variable and drop from feature set
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        dep_var = data['SP_DIRECTION']

        # -- All variables + clean:
        vars_for_logit = data[feature]

        # -- Run Logistic Regression model 1:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(vars_for_logit, dep_var, test_size=0.3, random_state=0)
        logit_model = LogisticRegression()
        logit_model.fit(X_train1, y_train1)
        pred = logit_model.predict(X_test1)  # predcition
        accuracy = logit_model.score(X_test1, y_test1) # Return the mean accuracy on the given test data and labels.
        prob = logit_model.predict_proba(X_test1) #	Probability estimates.
        df['ACCURACY'].append(accuracy)
        return df
'''

##########################################
# 3. Linear Regression: Predict Stock price
##########################################


