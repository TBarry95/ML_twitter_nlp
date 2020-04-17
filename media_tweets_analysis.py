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

# -- plot top 10 words:
'''df_top_words = pd.DataFrame({"WORD": top_words[0], "COUNT": top_words[1]})
plt.figure()
plt.bar(df_top_words["WORD"][0:10], df_top_words["COUNT"][0:10])
plt.xlabel('Words', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.title("Top 10 Words", fontsize=20)'''

##########################################################################
# Analysis:
# -- LDA?
# 1. Get Sentiment: Lexicon-based polarity
# 2. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 3. Logistic Regression: Predict Stock price Direction
##########################################################################

# LDA
# lda_output = fns.lda_model(df_all_tweets, 5, 15)

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
# 2. Logistic Regression: Predict Stock price Direction
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
out1 = logistic_regression(df_features, ['TWEET_COUNT_DAY', 'LEN_TWEET_SUM', 'MEAN_SENT1_PCT'])

##########################################
# 4. Linear Regression: Predict Stock price
##########################################

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


