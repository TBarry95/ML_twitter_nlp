# Des: Analysis of tweets extracted from 65 twitter media pages (150k+ tweets).
#      Similarily, goal is to predict stock price direction
#      using sentiment analysis, correlation matrix, and logistic regression.
# By: Tiernan Barry - x19141840 (NCI) - Data Mining and Machine Learning

# Libraries:
import Twitter_API_Module as twt
import numpy as np
import pandas as pd
import missingno as msno
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *
from matplotlib.pyplot import ion
import pandas as pd
ion() # enables interactive mode

# Source files (functions):
import functions_nlp as fns

##########################################################################
# A. EXTRACT:
##########################################################################
print("# -- Read in cleaned datasets -- #")
df_all_tweets = pd.read_csv("./cleaned_media_tweets.csv")
label_tweet_smaller = pd.read_csv("./label_tweet_smaller.csv")
gspc_df = pd.read_csv("./gspc_df.csv")
print("##########################################################")
print("##########################################################")

##########################################
# 3. Get Sentiment: Lexicon-based polarity
##########################################
print("# -- Apply 2 x Lexicon-Based Sentiment analysis:  -- #")
print("# -- 1.  Continuous: 1:-1 -- #")
df_all_tweets["SENTIMENT_CONT"] = fns.get_sentiment_lex_cont(df_all_tweets['PROCESSED_TEXT'])
print(df_all_tweets["SENTIMENT_CONT"].head())

print("# -- 2. Categorical: 1, 0, -1 -- #")
df_all_tweets["SENTIMENT_CATG"] = [1 if i > 0 else -1 for i in df_all_tweets["SENTIMENT_CONT"]]
print(df_all_tweets["SENTIMENT_CATG"].head())
print("##########################################################")
print("##########################################################")
##########################################
# 4. Get Sentiment: NB Classifier over tweets
##########################################
print("# -- Apply 2 x Naive Bayes Classifiers: Sentiment analysis:  -- #")
print("# -- 1. Multinomial Naive Bayes  -- #")
print("# -- 2. Bernoulli Naive Bayes  -- #")

# -- Train Multinomial NB on Twitter dataset from Kaggle:
nb_train, nb_test, nb_train_sent, nb_test_sent = train_test_split(label_tweet_smaller['PROCESSED_TEXT'], label_tweet_smaller['sentiment'], test_size=0.3, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(nb_train)
X_test_counts = count_vect.transform(nb_test)
tweets_counts = count_vect.transform([str(i) for i in df_all_tweets["PROCESSED_TEXT"]]) # ensuring all string formatted

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
pred_nb_all = nb.predict(tweets_counts_tfidf)

bn = BernoulliNB()
bn.fit(X_train_tfidf, nb_train_sent)
pred_bn = bn.predict(X_test_tfidf)
pred_bn_all = bn.predict(tweets_counts_tfidf)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# -- test lexicon: manually run check to see how many are correctly labelled
lex_test_cont = fns.get_sentiment_lex_cont(nb_test)
lex_test =  [1 if i > 0 else -1 for i in lex_test_cont]
# -- find how many are correct:
true = []
false = []
for i,j in zip(lex_test,nb_test_sent):
    if i == j:
        true.append(True)
    else:
        false.append(False)

print("##########################################################")
print("##########################################################")
print("# -- Evaluation of classification models: Accuracy tests: -- #")
print("Accuracy of Lexicon-Based Sentiment Classifier:", len(true)/len(nb_test))
print("Accuracy of Multinomial Naive Bayes Classifier:", nb.score(X_test_tfidf, nb_test_sent))
print("Accuracy of Bernoulli Naive Bayes Classifier:", bn.score(X_test_tfidf, nb_test_sent))
print("Applying Bernoulli NB as a predictor variable for stock prices")
print("Bernoulli NB Report:")
print(metrics.classification_report(pred_bn, nb_test_sent))
print("Bernoulli NB Confusion Matrix:")
print(metrics.confusion_matrix(pred_bn, nb_test_sent))
print("##########################################################")
print("##########################################################")
print("Multinomial  NB Report:")
print(metrics.classification_report(pred_nb, nb_test_sent))
print("Multinomial  NB Confusion Matrix:")
print(metrics.confusion_matrix(pred_nb, nb_test_sent))
# Cant verify if right or wrong, but assuming 77% right
print("##########################################################")
print("##########################################################")

# -- NB feature for predicitve analysis:
df_all_tweets['NB_SENTIMENT'] = pred_bn_all

##########################################
# 5. Get feature set: Aggregate tweets by date:
##########################################

print("# -- Aggregate Sentiment Data by Date: Build Feature set for Predictive Analysis -- #")

# Aggregate tweet data and sentiment data by date using Mean transformation
df_features = pd.DataFrame()
df_features['MEAN_SENT1'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_CATG'].mean()
df_features['MEAN_SENT2'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_CONT'].mean()
df_features['MEAN_SENT3_NB'] = df_all_tweets.groupby('DATE_TIME')['NB_SENTIMENT'].mean()
df_features['MEAN_SENT1_PCT'] = df_features['MEAN_SENT1'].pct_change()
df_features['MEAN_SENT2_PCT'] = df_features['MEAN_SENT2'].pct_change()
df_features['MEAN_SENT3_NB_PCT'] = df_features['MEAN_SENT3_NB'].pct_change()
df_features['FAV_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['FAV_COUNT'].mean()
df_features['RT_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['RT_COUNT'].mean()
df_features['TWEET_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_CATG'].count()
df_features['LEN_TWEET_SUM'] = df_all_tweets.groupby('DATE_TIME')['LEN_TWEET'].mean()
df_features['FOLLOWERS'] = df_all_tweets.groupby('DATE_TIME')['FOLLOWERS'].mean()

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
boxplot1 = df_features.boxplot(column=['RT_COUNT_DAY', 'FAV_COUNT_DAY'])
print("##########################################################")
print("##########################################################")
