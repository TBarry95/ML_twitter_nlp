



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















