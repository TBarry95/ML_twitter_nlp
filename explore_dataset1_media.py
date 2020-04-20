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
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *
from tabulate import tabulate

# Source files (functions):
import functions_nlp as fns

##########################################################################
# A. EXTRACT:
##########################################################################

# -- Read in tweets as sourced from get_datasets.py
df_all_tweets = pd.read_csv(r".\twitter_mass_media_data_2.csv")
gspc_df = pd.read_csv(r".\raw_datasets\^GSPC.csv")
gspc_df['pct_change'] = gspc_df['Close'].pct_change()
gspc_df['direction'] = [1 if i > 0 else 0 for i in gspc_df['pct_change']]

# -- Read in labelled tweets for training NB: taken from https://www.kaggle.com/kazanova/sentiment140
labelled_tweets = pd.read_csv(r".\raw_datasets\training.1600000.processed.noemoticon.csv", encoding='latin-1')

print("Raw datasets read in successfully:")
print("##########################################################")
print("##########################################################")
print("# -- Data overview report: -- #")
print("1. Media tweets: ")
print("Number of columns in raw dataset: ", len(df_all_tweets.columns))
print("Number of rows in raw dataset: ", len(df_all_tweets))
print("2. Labelled Tweets from Kaggle: ")
print("Number of columns in raw dataset: ", len(labelled_tweets.columns))
print("Number of rows in raw dataset: ", len(labelled_tweets))
print("3. S&P500 data")
print("Number of columns in raw dataset: ", len(gspc_df.columns))
print("Number of rows in raw dataset: ", len(gspc_df))
##########################################################################
# B. TRANSFORM:
##########################################################################

##########################################
# 1. Clean Tweets from tweepy
##########################################

print("# -- Checking Null Values: -- #")
msno.matrix(df_all_tweets, figsize= (50,30))
print("Dropping 2 sparsely populated columns ")
print("1. Reply to User ID ")
print("2. Reply to User ")
print("No important features are lost or affected for this analysis by dropping")
print("##########################################################")
print("##########################################################")

# -- Format date:
df_all_tweets['DATE_TIME'] = [str(i)[0:10] for i in df_all_tweets['DATE_TIME']]

# -- Make new column for processed name:
df_all_tweets['PROCESSED_TEXT'] = df_all_tweets['FULL_TEXT'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# -- Remove stop words:
df_all_tweets['PROCESSED_TEXT'] = [i for i in df_all_tweets['PROCESSED_TEXT'] if i not in stopwords.words('english')]

# -- Check for formatting:
word_cloud = fns.get_wordcloud(df_all_tweets, r"C:\Users\btier\Documents\news_word_cloud.png")

# -- bag of words - stop words already removed:
top_words = fns.get_top_words(df_all_tweets)

##########################################
# 2. Clean Tweets from Kaggle (only for training!)
##########################################

# -- Reduce columns:
labelled_tweets = labelled_tweets[['sentiment', 'text']]

# -- Reduce dataset for training (too big for comptuer):
label_0 = labelled_tweets[labelled_tweets['sentiment'] == 0]
label_4 = labelled_tweets[labelled_tweets['sentiment'] == 4]
label_0['sentiment'] = [-1 for i in label_0['sentiment']]
label_4['sentiment'] = [1 for i in label_4['sentiment']]
df = pd.DataFrame()
label_tweet_smaller = df.append(label_0[0:int(len(label_0)/5)])
label_tweet_smaller = label_tweet_smaller.append(label_4[0:int(len(label_4)/5)])

# -- Remove stop words:
label_tweet_smaller['PROCESSED_TEXT'] = label_tweet_smaller['text'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
label_tweet_smaller['PROCESSED_TEXT'] = [i for i in label_tweet_smaller['PROCESSED_TEXT'] if i not in stopwords.words('english')]

print("# -- Tweet cleaning for NLP: Both Media Tweets and Training dataset from Kaggle -- # ")
print("1. Removed all punctuation and special characters")
print("2. Removed all domain specific jargon such as RT (retweet)")
print("3. Removed all Stop Words")
print("##########################################################")
print("##########################################################")
##########################################
# 3. Get Sentiment: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1,0,1):
df_all_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in df_all_tweets["PROCESSED_TEXT"]])
df_all_tweets = fns.get_sentiment_pa(df_all_tweets)

##########################################
# 4. Get Sentiment: NB Classifier over tweets
##########################################

# -- Train Multinomial NB on Twitter dataset from Kaggle:
nb_train, nb_test, nb_train_sent, nb_test_sent = train_test_split(label_tweet_smaller['PROCESSED_TEXT'], label_tweet_smaller['sentiment'], test_size=0.3, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(nb_train)
X_test_counts = count_vect.transform(nb_test)
tweets_counts = count_vect.transform(df_all_tweets["PROCESSED_TEXT"])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
tweets_counts_tfidf = tfidf_transformer.transform(tweets_counts)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
nb = MultinomialNB()
nb.fit(X_train_tfidf, nb_train_sent)
pred_nb = nb.predict(X_test_tfidf)

bn = BernoulliNB()
bn.fit(X_train_tfidf, nb_train_sent)
pred_bn = bn.predict(X_test_tfidf)
from sklearn import metrics

print("Accuracy of Multinomial Naive Bayes Classifier:", nb.score(X_test_tfidf, nb_test_sent))
print("Accuracy of Bernoulli Naive Bayes Classifier:", bn.score(X_test_tfidf, nb_test_sent))
print("Applying Bernoulli NB as a predictor variable for stock prices")
print("Bernoulli NB Report:")
print(metrics.classification_report(pred_bn, nb_test_sent))
print("Bernoulli NB Confusion Matrix:")
print(metrics.confusion_matrix(pred_bn, nb_test_sent))
print("##########################################################")
print("##########################################################")

# -- Get sentiment score for tweets from media companies:
df_all_tweets["NB_SENTIMENT"] = nb.predict(tweets_counts_tfidf)

# Cant verify if right or wrong, but assuming 77% right

##########################################
# 5. Get feature set: Aggregate tweets by date:
##########################################

df_features = pd.DataFrame()
df_features['MEAN_SENT1'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].mean()
df_features['MEAN_SENT2'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_PA'].mean()
df_features['MEAN_SENT3_NB'] = df_all_tweets.groupby('DATE_TIME')['NB_SENTIMENT'].mean()
df_features['MEAN_SENT1_PCT'] = df_features['MEAN_SENT1'].pct_change()
df_features['MEAN_SENT2_PCT'] = df_features['MEAN_SENT2'].pct_change()
df_features['MEAN_SENT3_NB_PCT'] = df_features['MEAN_SENT3_NB'].pct_change()
df_features['FAV_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['FAV_COUNT'].sum()
df_features['RT_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['RT_COUNT'].sum()
df_features['TWEET_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].count()
df_features['LEN_TWEET_SUM'] = df_all_tweets.groupby('DATE_TIME')['LEN_TWEET'].sum()
df_features['FOLLOWERS'] = df_all_tweets.groupby('DATE_TIME')['FOLLOWERS'].sum()

# -- Handle infs:
df_features['MEAN_SENT1_PCT'][df_features['MEAN_SENT1_PCT'].values == -np.inf] = -0.99 # replace + and - infinity
df_features['MEAN_SENT1_PCT'][df_features['MEAN_SENT1_PCT'].values == np.inf] = 0.99
df_features['MEAN_SENT2_PCT'][df_features['MEAN_SENT2_PCT'].values == np.inf] = 0.99
df_features['MEAN_SENT2_PCT'][df_features['MEAN_SENT2_PCT'].values == -np.inf] = -0.99
df_features['MEAN_SENT3_NB_PCT'][df_features['MEAN_SENT3_NB_PCT'].values == -np.inf] = -0.99
df_features['MEAN_SENT3_NB_PCT'][df_features['MEAN_SENT3_NB_PCT'].values == np.inf] = 0.99

# -- Join tweets to stock prices:
gspc_df_features = gspc_df[['Date', 'Close', 'pct_change', 'direction']]
df_features = pd.merge(df_features, gspc_df_features, how='left', left_on='DATE_TIME', right_on='Date')
msno.matrix(df_features, figsize= (50,30))
df_features = df_features.dropna()

df_features.to_csv(r".\media_data_cleaned.csv", index=False)

from tabulate import tabulate
print("# -- Cleaned Data Set summary: Random Sample out of entire dataset (too big to print) -- #")
print(tabulate(df_features.describe(), headers=df_features.columns))
print("##########################################################")
print("##########################################################")
np.random.seed(0)
print("Boxplot between 3 sentiment aggregations")
print("Outliers are as a result of low day counts for each senntiment")
boxplot = df_features.boxplot(column=['MEAN_SENT1_PCT',
                                'MEAN_SENT2_PCT', 'MEAN_SENT3_NB_PCT'])
boxplot = df_features.boxplot(column=['RT_COUNT_DAY', 'FAV_COUNT_DAY'])
print("##########################################################")
print("##########################################################")
