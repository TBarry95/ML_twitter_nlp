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

# -- Read in labelled tweets for training NB: taken from https://www.kaggle.com/kazanova/sentiment140
labelled_tweets = pd.read_csv(r".\raw_datasets\training.1600000.processed.noemoticon.csv", encoding='latin-1')

print("Raw datasets read in successfully:")
print("##########################################################")
print("##########################################################")
print("# -- Data overview report: -- #")
print("1. Trump tweets: ")
print("Number of columns in raw dataset: ", len(trump_tweets.columns))
print("Number of rows in raw dataset: ", len(trump_tweets))
print("Date range of raw dataset: ", trump_tweets['DATE_TIME'][-1:].values, "to ", trump_tweets['DATE_TIME'][:1].values)
print("2. S&P500 prices")
print("Number of columns in raw dataset: ", len(gspc_df.columns))
print("Number of rows in raw dataset: ", len(gspc_df))
print("Date range of raw dataset: ", gspc_df['Date'][-1:].values, "to ", gspc_df['Date'][:1].values)
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

missing_val_summary = trump_tweets.isna().mean()
missing_val_summary = pd.DataFrame(missing_val_summary)
missing_val_summary = missing_val_summary.reset_index()
missing_val_summary.columns = ['FIELD', 'MEAN']
missing_val_summary.plot(kind='bar')
missing_val_param = 0.20
new_data_cols = missing_val_summary['FIELD'][missing_val_summary['MEAN'] <= missing_val_param]

# -- Reduce columns:
trump_tweets = trump_tweets[new_data_cols.values]
print("# -- Checking Null Values: -- #")
print("Dropped all columns which have more than", missing_val_param*100, "% missing values")
print("For Trump tweets, all columns are populated and no columns are dropped")

# -- Deal with NA values: Back fill followed by forward fill
msno.matrix(trump_tweets, figsize= (50,30))
trump_tweets = trump_tweets.fillna(method='bfill')
trump_tweets = trump_tweets.fillna(method='ffill')
msno.matrix(trump_tweets, figsize= (50,30))
print("##########################################################")
print("##########################################################")

# merge
trump_tweets = pd.merge(trump_tweets, spx, how='left', left_on='DATE_TIME', right_on='DATE')
del trump_tweets['CREATED_AT']
del trump_tweets['DATE']
print("# -- Trump tweets: Before processing -- #")
print(trump_tweets['FULL_TEXT'].head())
print("##########################################################")
print("##########################################################")
# -- Make new column for processed name:
trump_tweets['PROCESSED_TEXT'] = trump_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)|(^RT )|(\d+)", '', i))

# -- Check for formatting using word cloud:
word_cloud = fns.get_wordcloud(trump_tweets, r".\trump_word_cloud.png")

print("# -- Trump tweets: After processing -- #")
# -- Remove stop words:
trump_tweets['PROCESSED_TEXT'] = [i.lower() for i in trump_tweets['PROCESSED_TEXT'] if i not in stopwords.words('english')]
print(trump_tweets['PROCESSED_TEXT'].head())
print("##########################################################")
print("##########################################################")
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

trump_tweets.to_csv("./cleaned_trump_tweets.csv", index=False)
#label_tweet_smaller.to_csv("./label_tweet_smaller.csv", index=False)
#gspc_df.to_csv("./gspc_df.csv")
