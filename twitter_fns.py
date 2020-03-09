from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_credentials
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import Twitter_API_Module as twt

def get_tweet_pgs(user_name, num_pgs):
    tweets = twt.TwitterClientClass(twit_user=user_name).get_timeline_pages(num_pgs)
    tweetslist = []
    tweets_df = pd.DataFrame()
    for i in tweets:
        tweetslist.append(twt.AnalyseTweetsClass().dataframe_tweets(i))
    return tweetslist

# cnn = get_tweet_pgs('CNN', 200)

# df = pd.DataFrame()

def get_tweets_hashtag(num_pgs, hashtag):
    tweets = twt.TwitterClientClass().get_hashtag_tweets(num_pgs, hashtag)
    tweetslist = []
    for i in tweets:
        tweetslist.append(twt.AnalyseTweetsClass().dataframe_tweets(i))
    return tweetslist

# y = get_tweets_hashtag(1, '#corona')
