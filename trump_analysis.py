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

##########################################################################
# Extract:
##########################################################################

# -- Read in Trump tweets:
trump_tweets = pd.read_csv(r"C:\Users\btier\Documents\trump_tweets.csv")
trump_tweets.columns = ['SOURCE', 'FULL_TEXT', 'CREATED_AT', 'RETWEET_COUNT', 'FAV_COUNT', 'IS_RETWEET', 'ID_STR']
trump_tweets['CREATED_AT'] = [str(i)[0:10] for i in trump_tweets['CREATED_AT']]

# -- Read in SP500 price:
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")

# -- Get % change in daily prices, and boolean column of UP or DOWN
gspc_df['pct_change'] = [i*100 for i in gspc_df['Close'].pct_change()]
gspc_df['movement'] = ["UP" if i > 0 else "DOWN" for i in gspc_df['pct_change']]

# -- Merge datasets by date:
trump_tweets['DATE_TIME'] = [str(i)[6:10] + '-' + str(i)[0:2] + str(i)[2:5] for i in trump_tweets['CREATED_AT']]
spx = pd.DataFrame({"DATE": gspc_df['Date'], "SP_CLOSE": gspc_df['Close'], "SP_PCT_CHANGE": gspc_df['pct_change'], "SP_DIRECTION": gspc_df['movement']})
trump_tweets = pd.merge(trump_tweets, spx, how='left', left_on='DATE_TIME', right_on='DATE')

del trump_tweets['CREATED_AT']
del trump_tweets['DATE']

##########################################################################
# Transform:
##########################################################################

# -- Deal with NA values: Back fill followed by forward fill
msno.matrix(trump_tweets, figsize= (50,30))
trump_tweets = trump_tweets.fillna(method='bfill')
trump_tweets = trump_tweets.fillna(method='ffill')
msno.matrix(trump_tweets, figsize= (50,30))

# -- Make new column for processed name:
trump_tweets['PROCESSED_TEXT'] = trump_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
trump_tweets.to_csv(r"C:\Users\btier\Documents\trump_processed.csv", index=False)

# -- Check for formatting using word cloud:
word_cloud = fns.get_wordcloud(trump_tweets, r"C:\Users\btier\Documents\trump_word_cloud.png")

# -- Plot top words - removed stop words:
top_words = fns.get_top_words(trump_tweets)
df_top_words = pd.DataFrame({"WORD": top_words[0], "COUNT": top_words[1]})
plt.figure()
plt.bar(df_top_words["WORD"][0:10], df_top_words["COUNT"][0:10])
plt.xlabel('Words', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title("Top 10 Words", fontsize=20)

##########################################################################
# Analysis:
# LDA: Latent Dirichlet Allocation
# Sentiment Analysis: Lexicon
# Sentiment Analysis: ML Naive Bayes
#
##########################################################################

##########################################
# 1. LDA: Latent Dirichlet Allocation
##########################################

# -- K-Means clustering to find optimal K:


# -- LDA:
lda_output = fns.lda_model(trump_tweets, 5, 15)

##########################################
# 2. Sentiment Analysis: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1,0,1):
trump_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in trump_tweets["PROCESSED_TEXT"]])

# -- Lexicon-based sentiment (-1:1):
trump_tweets = fns.get_sentiment_pa(trump_tweets)

# -- Plot:
trump_tweets["SENTIMENT_1"].plot(kind='hist', legend=True)
trump_tweets["SENTIMENT_PA"].plot(kind='hist', legend=True)

# -- Get Average Sentiment for each day: Map/reduce?
df_feature_1 = pd.DataFrame()

'''
# -- Mapper: Key = date, value = sent.
for sent in range(0,len(trump_tweets['SENTIMENT_1'])):
    if trump_tweets['SENTIMENT_1'][sent] < 0:
        value = -1
        df_feature_1['SENTIMENT_1'].append(value)
    elif trump_tweets['SENTIMENT_1'][sent] > 0:
        value = 1
        df_feature_1['SENTIMENT_1'].append(value)
    elif trump_tweets['SENTIMENT_1'][sent] == 0:
        value = 0
        df_feature_1['SENTIMENT_1'].append(value)
    df_feature_1['DATE_TIME'].append(key)
    key = trump_tweets['DATE_TIME'][sent]
    print(('%s\t%s') % (key, value))

# -- Reducer:
last_date_key = None
aggregate_sentiment = 0
count_per_date = 0

for sentiment in sys.stdin:
    sentiment = sentiment.strip()  # if whitespace - removes
    this_date_key, sentiment_value = sentiment.split()  # splits mapper by tab escaped
    sentiment_value = float(sentiment_value)

    if last_date_key == this_date_key:
        count_per_date += 1
        aggregate_sentiment += sentiment_value
    else:
        if last_date_key:
            print(('%s\t%s\t%s') % (last_date_key, aggregate_sentiment / count_per_date, count_per_date))
        aggregate_sentiment = sentiment_value
        last_date_key = this_date_key
        count_per_date = 1

# -- Output the least popular / min count sentiment sentiment
if last_date_key == this_date_key:
    print(('%s\t%s\t%s') % (last_date_key, aggregate_sentiment / count_per_date, count_per_date))

df_1 = [i for i in trump_tweets["DATE_TIME"]]
for i in trump_tweets["SENTIMENT_1"]:
    df_1.append(i)

date = df_1[:int((len(df_1))/2)]
text = df_1[int((len(df_1))/2):]


df_pa = [i for i in trump_tweets["DATE_TIME"]]
for i in trump_tweets["SENTIMENT_PA"]:
    df_pa.append(i)

'''
##########################################
# 3. Sentiment Analysis: Naive Bayes Classifier
##########################################

# -- Train data: Dataset of movie reviews:


##########################################
# 4. Logistic Regression Model:
##########################################
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import _logistic_loss

ind_vars_for_logit = trump_tweets[['SENTIMENT_1','SENTIMENT_PA']]
ind_vars_for_logit2 = trump_tweets[['SENTIMENT_PA']]
dep_var = trump_tweets['SP_DIRECTION']

X_train, X_test, y_train, y_test = train_test_split(ind_vars_for_logit2, dep_var, test_size=0.3, random_state=0)

logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)
pred = logit_model.predict(X_test) # results are poor due to skewed data....

# create training set of balanced UP and DOWN
plt.figure()
num_up = len(trump_tweets['SP_DIRECTION'][trump_tweets['SP_DIRECTION'] == "UP"])
num_down = len(trump_tweets['SP_DIRECTION'][trump_tweets['SP_DIRECTION'] == "DOWN"])

##########################################
# 4. Correlation with Stock Market:
##########################################

import seaborn as sns

# -- Plot: Correlation Matrix Plot:
corr_mx = trump_tweets[['RETWEET_COUNT', 'FAV_COUNT','SP_CLOSE','SP_PCT_CHANGE','SENTIMENT_1', 'SENTIMENT_PA']].corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})





