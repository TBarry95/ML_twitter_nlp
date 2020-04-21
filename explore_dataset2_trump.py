# Des:
#
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
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
# Source files (functions):
import functions_nlp as fns

##########################################################################
# A. EXTRACT:
##########################################################################

# -- Read in Trump tweets:
trump_tweets = pd.read_csv(r".\trump_tweets.csv")
trump_tweets.columns = ['SOURCE', 'FULL_TEXT', 'CREATED_AT', 'RETWEET_COUNT', 'FAV_COUNT', 'IS_RETWEET', 'ID_STR']
trump_tweets['CREATED_AT'] = [str(i)[0:10] for i in trump_tweets['CREATED_AT']]

# -- Read in SP500 price:
gspc_df = pd.read_csv(r".\raw_datasets\^GSPC.csv")
gspc_df['pct_change'] = [i*100 for i in gspc_df['Close'].pct_change()]
gspc_df['movement'] = [1 if i > 0 else 0 for i in gspc_df['pct_change']]

# -- Merge datasets by date:
trump_tweets['DATE_TIME'] = [str(i)[6:10] + '-' + str(i)[0:2] + str(i)[2:5] for i in trump_tweets['CREATED_AT']]
spx = pd.DataFrame({"DATE": gspc_df['Date'], "SP_CLOSE": gspc_df['Close'], "SP_PCT_CHANGE": gspc_df['pct_change'], "SP_DIRECTION": gspc_df['movement']})
trump_tweets = pd.merge(trump_tweets, spx, how='left', left_on='DATE_TIME', right_on='DATE')

del trump_tweets['CREATED_AT']
del trump_tweets['DATE']

# -- Read in labelled tweets for training NB: taken from https://www.kaggle.com/kazanova/sentiment140
labelled_tweets = pd.read_csv(r".\raw_datasets\training.1600000.processed.noemoticon.csv", encoding='latin-1')

print("Raw datasets read in successfully:")
print("##########################################################")
print("##########################################################")
print("# -- Data overview report: -- #")
print("1. Trump tweets: ")
print("Number of columns in raw dataset: ", len(trump_tweets.columns))
print("Number of rows in raw dataset: ", len(trump_tweets))
print("2. S&P500 prices")
print("Number of columns in raw dataset: ", len(gspc_df.columns))
print("Number of rows in raw dataset: ", len(gspc_df))
print("3. Labelled Tweets from Kaggle: ")
print("Number of columns in raw dataset: ", len(labelled_tweets.columns))
print("Number of rows in raw dataset: ", len(labelled_tweets))
print("Merged Trump tweets with S&P500 stock prices")
print("##########################################################")
print("##########################################################")

##########################################################################
# B. TRANSFORM:
##########################################################################

##########################################
# Clean Trump dataset:
##########################################

# -- Deal with NA values: Back fill followed by forward fill
msno.matrix(trump_tweets, figsize= (50,30))
trump_tweets = trump_tweets.fillna(method='bfill')
trump_tweets = trump_tweets.fillna(method='ffill')
msno.matrix(trump_tweets, figsize= (50,30))
print("# -- Identify Missing Values: -- # ")
print("By joining 44k with 23k dataset, this results in missing values. Apply backward fill")
print("##########################################################")
print("##########################################################")

# -- Make new column for processed name:
trump_tweets['PROCESSED_TEXT'] = trump_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# -- Check for formatting using word cloud:
word_cloud = fns.get_wordcloud(trump_tweets, r".\trump_word_cloud.png")

# -- Remove stop words:
trump_tweets['PROCESSED_TEXT'] = [i for i in trump_tweets['PROCESSED_TEXT'] if i not in stopwords.words('english')]

##########################################
# Clean Tweets from Kaggle (only for training!)
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
label_tweet_smaller['PROCESSED_TEXT'] = label_tweet_smaller['text'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
label_tweet_smaller['PROCESSED_TEXT'] = [i for i in label_tweet_smaller['PROCESSED_TEXT'] if i not in stopwords.words('english')]

print("# -- Tweet cleaning for NLP: Both Trump dataset and Training dataset from Kaggle -- # ")
print("1. Removed all punctuation and special characters")
print("2. Removed all domain specific jargon such as RT (retweet)")
print("3. Removed all Stop Words")
print("##########################################################")
print("##########################################################")

##########################################
# 1. Get Sentiment: Lexicon-based polarity
##########################################
print("# -- Get sentiment for tweets: Lexicon based approach and Naive Bayes -- #")

# -- Lexicon-based sentiment (-1,0,1):
trump_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in trump_tweets["PROCESSED_TEXT"]])
trump_tweets = fns.get_sentiment_pa(trump_tweets)

##########################################
# Get Sentiment: NB Classifier over tweets
##########################################

# -- Train Multinomial NB on Twitter dataset from Kaggle:
nb_train, nb_test, nb_train_sent, nb_test_sent = train_test_split(label_tweet_smaller['PROCESSED_TEXT'], label_tweet_smaller['sentiment'], test_size=0.3, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(nb_train)
X_test_counts = count_vect.transform(nb_test)
tweets_counts = count_vect.transform(trump_tweets["PROCESSED_TEXT"])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
tweets_counts_tfidf = tfidf_transformer.transform(tweets_counts)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, nb_train_sent)
mnb.predict(X_test_tfidf)

bn = BernoulliNB()
bn.fit(X_train_counts, nb_train_sent)
pred_bn = bn.predict(X_test_tfidf)
from sklearn import metrics

print("Accuracy of Multinomial Naive Bayes Classifier:", mnb.score(X_test_tfidf, nb_test_sent))
print("Accuracy of Bernoulli Naive Bayes Classifier:", bn.score(X_test_tfidf, nb_test_sent))
print("Applying Bernoulli NB as a predictor variable for stock prices")
print("Bernoulli NB Report:")
print(metrics.classification_report(pred_bn, nb_test_sent))
print("Bernoulli NB Confusion Matrix:")
print(metrics.confusion_matrix(pred_bn, nb_test_sent))
print("##########################################################")
print("##########################################################")

# -- Get sentiment score for tweets from media companies:
trump_tweets["NB_SENTIMENT"] = bn.predict(tweets_counts_tfidf)

# Cant verify if right or wrong, but assuming 77% right
# -- Plot:
plt.figure()
trump_tweets["SENTIMENT_1"].plot(kind='hist', legend=True)
trump_tweets["SENTIMENT_PA"].plot(kind='hist', legend=True)
trump_tweets["NB_SENTIMENT"].plot(kind='hist', legend=True)

##########################################
#  Get feature set: Aggregate tweets by date:
##########################################

# -- Get Average Sentiment for each date
df_feature_set = pd.DataFrame()
sent_data_1 = trump_tweets.groupby('DATE_TIME')['SENTIMENT_1'].mean()
sent_data_2 = trump_tweets.groupby('DATE_TIME')['SENTIMENT_PA'].mean()
sent_data_3 = trump_tweets.groupby('DATE_TIME')['NB_SENTIMENT'].mean()
count_tweets_day = trump_tweets.groupby('DATE_TIME')['SENTIMENT_1'].count()
rt_per_day = trump_tweets.groupby('DATE_TIME')['RETWEET_COUNT'].sum()
fav_per_day = trump_tweets.groupby('DATE_TIME')['FAV_COUNT'].sum()
df_feature_set['DATE'] = [i for i in sent_data_1.index]
df_feature_set['MEAN_DAILY_SENT1'] = [i for i in sent_data_1]
df_feature_set['MEAN_DAILY_SENT2'] = [i for i in sent_data_2]
df_feature_set['MEAN_DAILY_SENT3_NB'] = [i for i in sent_data_3]
df_feature_set['MEAN_SENT3_NB_PCT'] = df_feature_set['MEAN_DAILY_SENT3_NB'].pct_change()
df_feature_set['PCT_CHG_SENT1'] = df_feature_set['MEAN_DAILY_SENT1'].pct_change()
df_feature_set['PCT_CHG_SENT2'] = df_feature_set['MEAN_DAILY_SENT2'].pct_change()
df_feature_set['DIRECTION1'] = [1 if i > 0 else 0 for i in df_feature_set['PCT_CHG_SENT1']]
df_feature_set['DIRECTION2'] = [1 if i > 0 else 0 for i in df_feature_set['PCT_CHG_SENT2']]
df_feature_set['DIRECTION3'] = [1 if i > 0 else 0 for i in df_feature_set['MEAN_SENT3_NB_PCT']]
df_feature_set['TWEET_COUNT'] = [i for i in count_tweets_day]
df_feature_set['FAV_COUNT'] = [i for i in fav_per_day]
df_feature_set['RT_COUNT'] = [i for i in rt_per_day]
df_feature_set = df_feature_set[len(df_feature_set)-1300:]

# -- Handle infs:
df_feature_set['PCT_CHG_SENT1'][df_feature_set['PCT_CHG_SENT1'].values == -np.inf] = -0.99 # replace + and - infinity
df_feature_set['PCT_CHG_SENT1'][df_feature_set['PCT_CHG_SENT1'].values == np.inf] = 0.99
df_feature_set['PCT_CHG_SENT2'][df_feature_set['PCT_CHG_SENT2'].values == np.inf] = 0.99
df_feature_set['PCT_CHG_SENT2'][df_feature_set['PCT_CHG_SENT2'].values == -np.inf] = -0.99
df_feature_set['MEAN_SENT3_NB_PCT'][df_feature_set['MEAN_SENT3_NB_PCT'].values == -np.inf] = -0.99
df_feature_set['MEAN_SENT3_NB_PCT'][df_feature_set['MEAN_SENT3_NB_PCT'].values == np.inf] = 0.99
# -- Join prices
all_data = pd.merge(df_feature_set, trump_tweets[['DATE_TIME', 'SP_CLOSE', 'SP_DIRECTION']], how='left', left_on='DATE', right_on='DATE_TIME')
all_data = all_data.drop_duplicates()

# -- Clean
all_data = all_data.dropna()
all_data.to_csv(r".\trump_data_cleaned.csv", index=False)

from tabulate import tabulate
print("# -- Cleaned Data Set summary: Random Sample out of entire dataset (too big to print) -- #")
print(tabulate(all_data.describe(), headers=all_data.columns))
print("##########################################################")
print("##########################################################")
np.random.seed(0)
print("Boxplot between 3 sentiment aggregations")
print("Outliers are as a result of low day counts for each senntiment")
boxplot = all_data.boxplot(column=['MEAN_DAILY_SENT1', 'MEAN_DAILY_SENT2', 'MEAN_DAILY_SENT3_NB'])
print("##########################################################")
print("##########################################################")