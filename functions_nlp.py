# pip install tweepy
# pip install alpha_vantage
# pip install quandl

from alpha_vantage.timeseries import TimeSeries
import alpha_vantage
import tweepy
import csv
import json
import quandl
import pandas as pd
import json
from pymongo import MongoClient
import re
# pip install wordcloud
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import Twitter_API_Module as twt
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
#%%time
from pyLDAvis import sklearn as sklearn_lda
from sklearn.cluster import KMeans
import pickle
import pyLDAvis
import os

# # # # # # # # # # # # #
# Extract:
# # # # # # # # # # # # #

def get_data_alpha_v(ticker):
    api_key = '1TKL74QWO8OFMHQM'
    ts = TimeSeries(key=api_key, output_format='json')
    raw_price_data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    ticker_dates = []
    ticker_close = []
    ticker_dict = []
    for i in raw_price_data.items():
        ticker_dict.append(i[1])
        ticker_dates.append(i[0])
    for i in ticker_dict:
        ticker_close.append(float(i.get('4. close')))
    df = pd.DataFrame()
    df['DATE'] = ticker_dates
    df['CLOSE_PX'] = ticker_close
    #ticker_raw_data = {"Date": ticker_dates, "Close Price": ticker_close}
    return df

def get_tweets_list(list_of_twitter_accs, num_pages):
    tweet_list = []
    for i in list_of_twitter_accs:
        tweet_list.append(twt.TwitterClientClass(twit_user=i).get_timeline_pages(num_pages))

    all = []
    for i in tweet_list:
        for ii in i:
            for iii in ii:
                all.append(iii)
    return all

def tweets_to_df(all):
    df_all_tweets = pd.DataFrame()
    df_all_tweets['TWEET_ID'] = [i.id for i in all]
    df_all_tweets['DATE_TIME'] = [str(i.created_at)[0:10] for i in all]
    df_all_tweets['TWITTER_ACC'] = [i.user.name for i in all]
    df_all_tweets['STR_ID'] = [i.id_str for i in all]
    df_all_tweets['FULL_TEXT'] = [i.full_text for i in all]
    df_all_tweets['HASHTAGS'] = [i.entities['hashtags'] for i in all]
    df_all_tweets['SOURCE'] = [i.source for i in all]
    df_all_tweets['FAV_COUNT'] = [i.favorite_count for i in all]
    df_all_tweets['RT_COUNT'] = [i.retweet_count for i in all]
    df_all_tweets['FOLLOWERS'] = [i.user.followers_count for i in all]
    df_all_tweets['TWEET_COUNT'] = [i.author.statuses_count for i in all]
    df_all_tweets['REPLY_TO_USER_ID'] = [i.in_reply_to_user_id for i in all]
    df_all_tweets['REPLY_TO_USER'] = [i.in_reply_to_screen_name for i in all]
    df_all_tweets['LEN_TWEET'] = [len(i) for i in df_all_tweets['FULL_TEXT']]
    df_all_tweets.sort_values(by='DATE_TIME', ascending=0)
    return df_all_tweets

def get_quandl(ticker):
    api_key = 'QK5pYuDbK7X6hZc9xj1x'
    quandl.ApiConfig.api_key = api_key
    data = quandl.get(ticker, authtoken=api_key)
    return data

def get_trump_json_data(local_filepath):
    with open(local_filepath, encoding="utf8") as json_trump_tweets:
        data = json.load(json_trump_tweets)
        return data

def get_json_data_to_df(data):
    df = pd.DataFrame()
    data_cols = [x for x in data[0]]
    df['SOURCE'] = [x.get('source') for x in data]
    df['TEXT'] = [x.get('text') for x in data]
    df['DATE']  = [x.get('created_at') for x in data]
    df['RETWEET_COUNT'] = [x.get('retweet_count') for x in data]
    df['FAVOURITE_COUNT'] = [x.get('favorite_count') for x in data]
    df['IS_RETWEETED'] = [x.get('is_retweet') for x in data]
    df['ID_STR'] = [x.get('id_str') for x in data]
    return df

def get_tweet_pgs(user_name, num_pgs):
    tweets = twt.TwitterClientClass(twit_user=user_name).get_timeline_pages(num_pgs)
    tweetslist = []
    tweets_df = pd.DataFrame()
    for i in tweets:
        tweetslist.append(twt.AnalyseTweetsClass().dataframe_tweets(i))
    return tweetslist

def get_tweets_hashtag(num_pgs, hashtag):
    tweets = twt.TwitterClientClass().get_hashtag_tweets(num_pgs, hashtag)
    tweetslist = []
    for i in tweets:
        tweetslist.append(twt.AnalyseTweetsClass().dataframe_tweets(i))
    return tweetslist


# # # # # # # # # # # # #
# Explore Data:
# # # # # # # # # # # # #

def lda_model(df_all_tweets, num_topics, num_words_per_topic):
    # start count vector with stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(df_all_tweets['PROCESSED_TEXT'])
    # Create / fit LDA
    lda = LDA(n_components=num_topics, n_jobs=-1)
    lda.fit(count_data)
    words = count_vectorizer.get_feature_names()
    topic_num = []
    topic_list = []
    for topic_id, topic in enumerate(lda.components_):
        topic_num.append("Topic #%d:" % topic_id)
        topic_list.append(" ".join([words[i] for i in topic.argsort()[:-num_words_per_topic - 1:-1]]))
    return pd.DataFrame({"TOPIC_ID":topic_num, "TOPIC_WORDS": topic_list})

def lda_vis(df_all_tweets, num_topics):
    lda_vis_path = r"C:\Users\btier\Documents\lda_vis.html"
    # start count vector with stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(df_all_tweets['PROCESSED_TEXT'])
    # Create / fit LDA
    lda = LDA(n_components=num_topics, n_jobs=-1)
    lda.fit(count_data)
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(lda_vis_path, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
    with open(lda_vis_path) as f:
        LDAvis_prepared = pickle.load(f)
    return pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_' + str(num_topics) + '.html')

def elbow_graph(list, df_all_tweets):
    num_cluster = [i for i in list]
    inertia = []
    count_data = count_vectorizer.fit_transform(df_all_tweets['PROCESSED_TEXT'])
    for i in list:
        km = KMeans(i).fit(count_data)
        inertia.append(km.inertia_)
    return pd.DataFrame({"CLUSTERS": num_cluster, "DISTORTION": inertia})

def clean_text_words1(trump_df):
    # trump_df_cln = trump_df.drop(
    #     columns=['source', 'created_at', 'retweet_count', 'favorite_count', 'is_retweet', 'id_str'], axis=1)
    trump_df_cln['PROCESSED_TEXT'] = trump_df_cln['FULL_TEXT'].map(lambda i: re.sub('[,\.!?"]', '', i))
    for i in trump_df_cln['PROCESSED_TEXT']:
        i.replace('"', '')
    #trump_df_cln = trump_df_cln.drop(columns=['text'])
    return (trump_df_cln)

def clean_text_words(trump_df_all):
    trump_df_cln = pd.DataFrame()
    trump_df_cln['SOURCE'] = trump_df_all['SOURCE']
    trump_df_cln['PROCESSED_TEXT'] = trump_df_all['FULL_TEXT'].map(lambda i: re.sub('[,\.!?""@“”]', '', i))
    trump_df_cln['DATE'] = tweet_date_format(trump_df_all)
    trump_df_cln['RETWEET_COUNT'] = trump_df_all['RETWEET_COUNT']
    trump_df_cln['FAVOURITE_COUNT'] = trump_df_all['FAVOURITE_COUNT']
    trump_df_cln['IS_RETWEETED'] = trump_df_all['IS_RETWEETED']
    trump_df_cln['ID_STR'] = trump_df_all['ID_STR']
    return (trump_df_cln)

def clean_text_words3(trump_df_all):
    trump_df_all['PROCESSED_TEXT'] = trump_df_all['FULL_TEXT'].map(lambda i: re.sub('[,\.!?""@“”]', '', i))
    #trump_df_all['DATE'] = tweet_date_format(trump_df_all)
    return (trump_df_all)

def tweet_date_format(trump_df_all):
    yr = []
    day = []
    mnt = []
    for i in trump_df_all['DATE']:
        yr.append(i[26:30])
        day.append(i[8:10])
        if i[4:7] == 'Jan':
            mnt.append('01')
        elif i[4:7] == 'Feb':
            mnt.append('02')
        elif i[4:7] == 'Mar':
            mnt.append('03')
        elif i[4:7] == 'Apr':
            mnt.append('04')
        elif i[4:7] == 'May':
            mnt.append('05')
        elif i[4:7] == 'Jun':
            mnt.append('06')
        elif i[4:7] == 'Jul':
            mnt.append('07')
        elif i[4:7] == 'Aug':
            mnt.append('08')
        elif i[4:7] == 'Sep':
            mnt.append('09')
        elif i[4:7] == 'Oct':
            mnt.append('10')
        elif i[4:7] == 'Nov':
            mnt.append('11')
        elif i[4:7] == 'Dec':
            mnt.append('12')
    df_dts = pd.DataFrame({"Year": yr, "Month": mnt, "Day": day})
    new_dt = []
    for i in range(0,len(yr)):
        new_dt.append("-".join([df_dts['Year'][i], df_dts['Month'][i], df_dts['Day'][i]]))
    return new_dt

def get_wordcloud(trump_df_cln, filepath):
    make_string = ','.join(list(trump_df_cln['PROCESSED_TEXT'].values))
    word_cloud_obj = WordCloud(background_color="white", width=550, height=550, max_words=100, contour_width=2, contour_color='steelblue')
    word_cloud_obj.generate(make_string)
    word_cloud_obj.to_file(filepath)
    return word_cloud_obj.generate(make_string)

def get_top_words(trump_df_clean):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(trump_df_clean['PROCESSED_TEXT'])
    single_words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(single_words))
    for i in count_data:
        total_counts += i.toarray()[0]
    count_dict = (zip(single_words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:50] # removing 'https',  'tco', 'rt',
    single_words = [i[0] for i in count_dict]
    counts = [i[1] for i in count_dict]
    return [single_words, counts]

def get_sentiment_lex_cont(trump_df_clean):
    sentiment = []
    for i in trump_df_clean:
        i = str(i)
        blob = TextBlob(i)
        sentiment.append(blob.sentiment.polarity)
    return sentiment

def get_sentiment_nbayes(trump_df_clean):
    sentiment = []
    '''extremely slow'''
    for i in trump_df_clean['PROCESSED_TEXT']:
        blob = TextBlob(i, analyzer = NaiveBayesAnalyzer())
        sentiment.append(blob.sentiment.classification)
    trump_df_clean['SENTIMENT_NB'] = sentiment
    return trump_df_clean

# # # # # # # # # # # # #
# Predictions:
# # # # # # # # # # # # #


