# pip install textblob
import Twitter_API_Module as twt
import numpy as np
import pandas as pd
import re
import missingno as msno
import functions_nlp as fns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Read in tweets as sourced from get_datasets.py
df_all_tweets = pd.read_csv(r"C:\Users\btier\Documents\twitter_mass_media_data.csv")
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")

# Make new column for processed name:
df_all_tweets['PROCESSED_TEXT'] = df_all_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# Check for formatting:
word_cloud = fns.get_wordcloud(df_all_tweets, r"C:\Users\btier\Documents\news_word_cloud.png")

# bag of words - stop words already removed:
top_words = fns.get_top_words(df_all_tweets)

# plot top 10 words:
df_top_words = pd.DataFrame({"WORD": top_words[0], "COUNT": top_words[1]})
plt.figure()
plt.bar(df_top_words["WORD"][0:10], df_top_words["COUNT"][0:10])
plt.xlabel('Words', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.title("Top 10 Words", fontsize=20)

# get sentiment: 2 sentiments Textblob: a: (1,0,-1), b: (1:-1)
df_all_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in df_all_tweets["PROCESSED_TEXT"]])
df_all_tweets = fns.get_sentiment_pa(df_all_tweets)

df_all_tweets['DATE_TIME'] = [str(i)[0:10] for i in df_all_tweets['DATE_TIME']]

df_all_tweets = pd.merge(df_all_tweets, pd.DataFrame({"DATE": gspc_df['Date'], "SP_CLOSE": gspc_df['Close']}), how='left', left_on='DATE_TIME', right_on='DATE')
msno.matrix(df_all_tweets, figsize= (50,30))

data_4_lr = pd.DataFrame({"FAV_COUNT": df_all_tweets['FAV_COUNT'],
                          "RT_COUNT": df_all_tweets['RT_COUNT'],
                          "FOLLOWERS": df_all_tweets['FOLLOWERS'],
                          "SENTIMENT_PA": df_all_tweets['SENTIMENT_PA'],
                          "SP_CLOSE": df_all_tweets['SP_CLOSE'].fillna(method='ffill')})
msno.matrix(data_4_lr, figsize= (50,30))








# LDA
lda_output = fns.lda_model(df_all_tweets, 5, 15)



