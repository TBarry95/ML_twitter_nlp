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
from matplotlib.pyplot import ion
from matplotlib.pyplot import plot
ion() # enables interactive mode
from wordcloud import WordCloud

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
print("Date range of raw dataset: ", df_all_tweets['DATE_TIME'][-1:].values, "to ", df_all_tweets['DATE_TIME'][:1].values)
print("2. Labelled Tweets from Kaggle: ")
print("Number of columns in raw dataset: ", len(labelled_tweets.columns))
print("Number of rows in raw dataset: ", len(labelled_tweets))
print("3. S&P500 data")
print("Number of columns in raw dataset: ", len(gspc_df.columns))
print("Number of rows in raw dataset: ", len(gspc_df))
print("Date range of raw dataset: ", gspc_df['Date'][-1:].values, "to ", gspc_df['Date'][:1].values)
print("##########################################################")
print("##########################################################")

# -- get % of missing values for each column:
missing_val_summary = df_all_tweets.isna().mean()
pc_missing = (sum(missing_val_summary) / len(missing_val_summary))*100
missing_val_summary = pd.DataFrame(missing_val_summary)
missing_val_summary.plot(kind='bar', legend=False, title="Proportion of Missing Values: Media Tweets")
missing_val_summary = missing_val_summary.reset_index()
missing_val_summary.columns = ['FIELD', 'MEAN']
missing_val_param = 0.2

new_data_cols = missing_val_summary['FIELD'][missing_val_summary['MEAN'] <= missing_val_param]
missing_data = missing_val_summary['FIELD'][missing_val_summary['MEAN'] > missing_val_param].values

# -- Reduce columns:
print("# -- Checking Null Values: -- #")
new_data = df_all_tweets[new_data_cols.values]
print("Dropped all columns which have more than", missing_val_param*100, "% missing values")
print("Columns dropped from tweet dataset: ")
print(missing_data)

##########################################################################
# B. TRANSFORM:
##########################################################################

##########################################
# 1. Clean Tweets from tweepy
##########################################
# .matrix(new_data, figsize= (50,30))
print("##########################################################")
print("##########################################################")
print("# -- Get word cloud before text processing -- #")
# -- word cloud before processing:
make_string = ','.join(list(new_data['FULL_TEXT'].values))
wc = WordCloud(background_color="white", width=550, height=550, max_words=100, contour_width=2,
                           contour_color='steelblue')
wc.generate(make_string)
wc.to_file(r".\media_wordcloud_before.png")
print("##########################################################")
print("##########################################################")

# -- Format date:
new_data['DATE_TIME'] = [str(i)[0:10] for i in new_data['DATE_TIME']]

print("# -- Print head of tweets: Check punctuation, Special characters -- #")
# -- Make new column for processed name:
print(new_data['FULL_TEXT'].head(10))

print("##########################################################")
print("##########################################################")

print("# -- Remove punctuation, numbers, and special characters using Regular Expressions -- #")
new_data['PROCESSED_TEXT'] = new_data['FULL_TEXT'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|( RT )|(^RT )|(\d+)", '', i))
print(new_data['PROCESSED_TEXT'].head(10))
print("##########################################################")
print("##########################################################")
print("# -- Remove stop words, apply lowercase, and export new word cloud after cleaning -- #")
# -- Remove stop words:
new_data['PROCESSED_TEXT'] = [i.lower() for i in new_data['PROCESSED_TEXT'] if i not in stopwords.words('english')]
print(new_data['PROCESSED_TEXT'].head(10))
print("##########################################################")
print("##########################################################")
# -- Check for formatting:
word_cloud = fns.get_wordcloud(new_data,r".\media_wordcloud_after.png")

# -- bag of words - stop words already removed:
top_words = fns.get_top_words(new_data)

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
label_tweet_smaller['PROCESSED_TEXT'] = label_tweet_smaller['text'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)|(^RT )|(\d+)", '', i))
label_tweet_smaller['PROCESSED_TEXT'] = [i for i in label_tweet_smaller['PROCESSED_TEXT'] if i not in stopwords.words('english')]

print("# -- Tweet cleaning for NLP: Both Media Tweets and Training dataset from Kaggle -- # ")
print("1. Removed all punctuation, numbers and special characters")
print("2. Removed all domain specific jargon such as RT (retweet)")
print("3. Removed all Stop Words")
print("##########################################################")
print("##########################################################")

print("# -- Export cleaned datasets to working directory -- #")
new_data.dropna()
label_tweet_smaller.dropna()

new_data.isna().mean()

new_data.to_csv("./cleaned_media_tweets.csv", index=False)
label_tweet_smaller.to_csv("./label_tweet_smaller.csv", index=False)
gspc_df.to_csv("./gspc_df.csv")
