# Des:
# By:
import functions_nlp as fns
import pandas as pd
# pip install wordcloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')

# # # # # # # # # # # # #
# Extract: - Write to MongoDB?
# # # # # # # # # # # # #

# Get stock prices:
spx_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("SPX"))
djia_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("DJIA"))
combined_df = pd.DataFrame(spx_df.append(djia_df))

# Get Trump Tweets:
# Includes RTs
trump_df_all = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\trump_tweets.json"))
potus_df_all = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\potus_tweets.json"))
# Excludes RTs
trump_df_nort = fns.get_json_data_to_df(fns.get_trump_json_data(r"C:\Users\btier\Documents\trump_tweets_nort.json"))

# # # # # # # # # # # # #
# Explore Data:
# # # # # # # # # # # # #

# Timeseries:

# LDA analysis:
# word cloud: with RT's included:
trump_df_cln = fns.get_word_cloud(trump_df_all, r"C:\Users\btier\Documents\trump_word_cloud.png")
# word cloud: with RT's excluded:
trump_df_nrt_cln = fns.get_word_cloud(trump_df_nort, r"C:\Users\btier\Documents\trump_nort_wordcloud.png")


# bag of words:

count_vectorizer = CountVectorizer(stop_words='english')

count_data = count_vectorizer.fit_transform(trump_df_nrt_cln['processed_text'])

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

    '''
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    return plt.show()'''

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





