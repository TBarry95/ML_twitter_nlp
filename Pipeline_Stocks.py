# Des:
# By:
# pip install wordcloud
import functions_nlp as fns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # #
# Extract: - Write to MongoDB?
# # # # # # # # # # # # #

# 1. Get stock prices:
# -- SPX
spx_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("SPX"))
spx_target = spx_df['CLOSE_PRICE'].shift(-1) # get dependent variable (-1 = predict 1 day ahead, etc)
spx_validate = spx_df[0:int(len(spx_df)*0.10)]
spx_target_validate = spx_target[0:int(len(spx_df)*0.10)]
spx_df_test_train = spx_df[int(len(spx_df)*0.10):] # this to be split between test and train
spx_target_test_train = spx_target[int(len(spx_df)*0.10):]

# -- DJIA
djia_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("DJIA"))
djia_target = djia_df['CLOSE_PRICE'].shift(-1)  # get dependent variable (-1 = predict 1 day ahead, etc)
djia_validate = djia_df[0:int(len(djia_df)*0.10)]
djia_target_validate = djia_target[0:int(len(djia_df)*0.10)]
djia_df_test_train = djia_df[int(len(djia_df)*0.10):] # this to be split between test and train
djia_target_test_train = djia_target[int(len(djia_target)*0.10):]


# 2. Get Trump Tweets:
# -- Includes RTs
trump_df_all = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\trump_tweets.json"))
potus_df_all = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\potus_tweets.json"))
# -- Excludes RTs
trump_df_nort = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\trump_tweets_nort.json"))

# # # # # # # # # # # # #
# Explore Data:
# # # # # # # # # # # # #

# Stock timeseries:

# Trump twitter data:
# -- Clean: Cleans trump_df_all['text'] column to 'processed text'
trump_df_clean = fns.clean_text_words(trump_df_all)
# ----- Ensure cleaned by checking Wordcloud:
word_cloud = fns.get_wordcloud(trump_df_clean, r"C:\Users\btier\Documents\trump_word_cloud.png")

# -- Bag of words:
# ----- Excludes top 3 (https, tco, rt)
top_words = fns.get_top_words(trump_df_clean)

# -- Sentiment: Add sentiment to trump_df_clean
trump_df_clean = fns.get_sentiment_pa(trump_df_clean)

# # # # # # # # # # # # #
# Process data:
# # # # # # # # # # # # #

# Split data for test and training: (validate already kept aside!)
# ---- SPLIT DATA: Train 70%, Test 30% - (of 90% - due to validation set)
spx_train, spx_test, spx_price_train, spx_price_test = train_test_split(spx_df_test_train, spx_target_test_train, test_size=0.3, shuffle=False)

djia_train, djia_test, djia_price_train, djia_price_test = train_test_split(djia_df_test_train, djia_target_test_train, test_size=0.3, shuffle=False)

correlations = np.abs(spx_train.corrwith(spx_price_train))
features =  list(correlations.sort_values(ascending=False)[0:10].index)







'''
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(trump_df_clean['processed_text'])

def plt_top_20_words(count, count_vect):
    single_words = count_vect.get_feature_names()
    total_counts = np.zeros(len(single_words))
    for i in count:
        total_counts += i.toarray()[0]

    count_dict = (zip(single_words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    single_words = [i[0] for i in count_dict]
    counts = [i[1] for i in count_dict]
    return [single_words, counts]


    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    return plt.show()

plt_top_20_words(count_data,count_vectorizer)[0]






o = plt_top_20_words(count_data, count_vectorizer)
plt.figure(2, figsize=(15, 15 / 1.6180))
plt.subplot(title='10 most common words')
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, o[1], palette='husl')
plt.xticks(x_pos, o[0], rotation=90)
plt.xlabel('words')
plt.ylabel('counts')
plt.show()
plt.



# # # # # # # # # # # # #
# Data Understanding
# # # # # # # # # # # # #











# cols_list = list(combined_df.columns.values)



# # # # # # # # # # # # #
# Transform:
# # # # # # # # # # # # #





'''