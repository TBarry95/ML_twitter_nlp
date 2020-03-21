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

###########################################
# Extract:
###########################################

twitter_pgs = ["CNN", "BBCWorld", "BBCBreaking", "BBCNews", "ABC", "itvnews", "Independent",
               "CBSNews", "MSNBC", "nytimes", "FT", "business", "cnni", "RT_com", "AJEnglish", "CBS", "NewsHour",
               "NPR", "BreakingNews", "cnnbrk", "WSJ", "Reuters", "SkyNews", "CBCAlerts"]

tweets_list = fns.get_tweets_list(twitter_pgs, 5) # change to 40 +

df_all_tweets = fns.tweets_to_df(tweets_list)

df_all_tweets = df_all_tweets.sort_values(by='DATE_TIME', ascending=0)

###########################################
# Transform:
###########################################

# Check for NAs
#msno.matrix(df_all_tweets)

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

# get sentiment: 3 sentiments Textblob: a: (1,0,-1), b: (1:-1), c: bayes classifier (pos,neg)
df_all_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in df_all_tweets["PROCESSED_TEXT"]])
df_all_tweets = fns.get_sentiment_pa(df_all_tweets)

# LDA
lda_output = fns.lda_model(df_all_tweets, 5, 15)



# Visualise :
# LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# fns.lda_vis(df_all_tweets, 5)

