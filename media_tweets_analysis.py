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
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import csv
import sys
nltk.download('stopwords')
nltk.download('punkt')
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.metrics import *

# Source files (functions):
import functions_nlp as fns

##########################################################################
# A. EXTRACT:
##########################################################################

# -- Read in tweets as sourced from get_datasets.py
df_all_tweets = pd.read_csv(r"C:\Users\btier\Documents\twitter_mass_media_data_2.csv")
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")
gspc_df['pct_change'] = gspc_df['Close'].pct_change()
gspc_df['direction'] = [1 if i > 0 else 0 for i in gspc_df['pct_change']]

# -- Read in labelled tweets for training NB: taken from https://www.kaggle.com/kazanova/sentiment140
labelled_tweets = pd.read_csv(r"C:\Users\btier\Downloads\training.1600000.processed.noemoticon.csv", encoding='latin-1')

##########################################################################
# B. TRANSFORM:
##########################################################################

##########################################
# 1. Clean Tweets from tweepy
##########################################
# -- Format date:
df_all_tweets['DATE_TIME'] = [str(i)[0:10] for i in df_all_tweets['DATE_TIME']]

# -- Make new column for processed name:
df_all_tweets['PROCESSED_TEXT'] = df_all_tweets['FULL_TEXT'].map(lambda i: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# -- Remove stop words:
df_all_tweets['PROCESSED_TEXT'] = [i for i in df_all_tweets['PROCESSED_TEXT'] if i not in stopwords.words('english')]

# -- Check for formatting:
word_cloud = fns.get_wordcloud(df_all_tweets, r"C:\Users\btier\Documents\news_word_cloud.png")

# -- bag of words - stop words already removed:
top_words = fns.get_top_words(df_all_tweets)

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
label_tweet_smaller['PROCESSED_TEXT'] = [i for i in label_tweet_smaller['text'] if i not in stopwords.words('english')]

##########################################
# 3. Get Sentiment: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1,0,1):
df_all_tweets["SENTIMENT_1"] = np.array([twt.AnalyseTweetsClass().sentiment_analyser(i) for i in df_all_tweets["PROCESSED_TEXT"]])
df_all_tweets = fns.get_sentiment_pa(df_all_tweets)

##########################################
# 4. Get Sentiment: NB Classifier over tweets
##########################################

# -- Train Multinomial NB on Twitter dataset from Kaggle:
nb_train, nb_test, nb_train_sent, nb_test_sent = train_test_split(label_tweet_smaller['PROCESSED_TEXT'], label_tweet_smaller['sentiment'], test_size=0.3, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(nb_train)
X_test_counts = count_vect.transform(nb_test)
tweets_counts = count_vect.transform(df_all_tweets["PROCESSED_TEXT"])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
tweets_counts_tfidf = tfidf_transformer.transform(tweets_counts)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
clf = nb.fit(X_train_tfidf, nb_train_sent)
nb.predict(X_test_tfidf)
print("Accuracy of Naive Bayes Classifier:", nb.score(X_test_tfidf, nb_test_sent))

# -- Get sentiment score for tweets from media companies:
df_all_tweets["NB_SENTIMENT"] = nb.predict(tweets_counts_tfidf)

# Cant verify if right or wrong, but assuming 77% right

##########################################
# 5. Get feature set: Aggregate tweets by date:
##########################################

df_features = pd.DataFrame()
df_features['MEAN_SENT1'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].mean()
df_features['MEAN_SENT2'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_PA'].mean()
df_features['MEAN_SENT3_NB'] = df_all_tweets.groupby('DATE_TIME')['NB_SENTIMENT'].mean()
df_features['MEAN_SENT1_PCT'] = df_features['MEAN_SENT1'].pct_change()
df_features['MEAN_SENT2_PCT'] = df_features['MEAN_SENT2'].pct_change()
df_features['MEAN_SENT3_NB_PCT'] = df_features['MEAN_SENT3_NB'].pct_change()
df_features['FAV_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['FAV_COUNT'].sum()
df_features['RT_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['RT_COUNT'].sum()
df_features['TWEET_COUNT_DAY'] = df_all_tweets.groupby('DATE_TIME')['SENTIMENT_1'].count()
df_features['LEN_TWEET_SUM'] = df_all_tweets.groupby('DATE_TIME')['LEN_TWEET'].sum()
df_features['FOLLOWERS'] = df_all_tweets.groupby('DATE_TIME')['FOLLOWERS'].sum()

# -- Handle infs:
df_features['MEAN_SENT1_PCT'][df_features['MEAN_SENT1_PCT'].values == -np.inf] = -0.99 # replace + and - infinity
df_features['MEAN_SENT1_PCT'][df_features['MEAN_SENT1_PCT'].values == np.inf] = 0.99
df_features['MEAN_SENT2_PCT'][df_features['MEAN_SENT2_PCT'].values == np.inf] = 0.99
df_features['MEAN_SENT2_PCT'][df_features['MEAN_SENT2_PCT'].values == -np.inf] = -0.99
df_features['MEAN_SENT3_NB_PCT'][df_features['MEAN_SENT3_NB_PCT'].values == -np.inf] = -0.99
df_features['MEAN_SENT3_NB_PCT'][df_features['MEAN_SENT3_NB_PCT'].values == np.inf] = 0.99

# -- Join tweets to stock prices:
gspc_df_features = gspc_df[['Date', 'Close', 'pct_change', 'direction']]
df_features = pd.merge(df_features, gspc_df_features, how='left', left_on='DATE_TIME', right_on='Date')
msno.matrix(df_features, figsize= (50,30))
df_features = df_features.dropna()

##########################################################################
# C. ANALYSIS:
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
# 2. Split data: Logistic regression and for Linear Regression
# 3. PCA
# 4. Logistic Regression: Predict Stock price Direction
# 5. Random Forest Regression: Predict Stock price
##########################################################################

##########################################
# 1. Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = df_features[['MEAN_SENT1', 'MEAN_SENT2', 'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
       'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
       'FOLLOWERS', 'Close']].corr()

mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print("# -- Correlation MAtrix Results: -- #")
print("No significant correlation between SP500 close and any tweet sentiment metrics (although positive")
print("##########################################################")
print("##########################################################")

##########################################
# 2. Split data: Logistic regression and for Linear Regression
##########################################

df_features = df_features.dropna()
df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()

df_features_ind = df_features[['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS']]

# --  Random Forest
data_train_lin, data_test_lin, price_train_lin, price_test_lin = train_test_split(df_features_ind, df_features['Close'], test_size=0.3, random_state=0)

# -- Logistic Regression + RF?
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(df_features_ind, df_features['direction'], test_size=0.3, random_state=0)

##########################################
# 3. PCA:
##########################################

# 1. Logistic regression:
# -- Initialise PCA class
pca = PCA()
data_reduced_train = pca.fit_transform(scale(data_train_log))
data_reduced_test = pca.transform(scale(data_test_log))

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')

df_pcs_train_log = pd.DataFrame(data=data_reduced_train, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])
df_pcs_test_log = pd.DataFrame(data=data_reduced_test, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])

# 2. RF regression:
# -- Initialise PCA class
pca1 = PCA()
data_reduced_train1 = pca1.fit_transform(scale(data_train_lin))
data_reduced_test1 = pca1.transform(scale(data_test_lin))

df_pcs_train_lin = pd.DataFrame(data=data_reduced_train1, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])
df_pcs_test_lin = pd.DataFrame(data=data_reduced_test1, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])

print("# -- PCA Results: Variance Explained per PC -- #")
print(variance_explained_2)
print("##########################################################")
print("##########################################################")

##########################################
# 4. Logistic Regression: Predict Stock price Direction
##########################################

# -- Get predicition using 5 PC's:
logit_model = LogisticRegression()
logit_model.fit(df_pcs_train_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_train_log)
pred = logit_model.predict(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])  # predcition
accuracy = logit_model.score(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],price_test_log) # Return the mean accuracy on the given test data and labels.
prob = logit_model.predict_proba(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]) #	Probability estimates.

# -- Find Metrics and Visualise:
print("# -- PCA Logistic Regression Test -- #")
print("Mean accuracy: ", accuracy)
# -- Print Equation:
intercept_log = logit_model.intercept_
coefs = logit_model.coef_
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
print("Logit 5 PC's = ", intercept_log[0], '+ (', pc[0],round(coefs[0][0],3), ') + (' ,pc[1],round(coefs[0][1],3), ') + (', pc[2],round(coefs[0][2],3),
      ') + (' ,pc[3],round(coefs[0][3],3), ') + (', pc[4],round(coefs[0][4],3), ')')
print("##########################################################")
print("##########################################################")
# -- Get predicition using X variables:
'''['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS'],'''

def logistic_regression(Xtrain, Xtest, ytrain, ytest, list_of_features):
    logit_model = LogisticRegression()
    logit_model.fit(Xtrain[list_of_features], ytrain)
    pred = logit_model.predict(Xtest[list_of_features])  # predcition
    accuracy = logit_model.score(Xtest[list_of_features], ytest) # Return the mean accuracy on the given test data and labels.
    prob = logit_model.predict_proba(Xtest[list_of_features]) #	Probability estimates.
    intercept_log = logit_model.intercept_
    coefs = logit_model.coef_
    return [pred, accuracy, prob, intercept_log, coefs, list_of_features]

log_out1 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log,
                               ['MEAN_SENT1', 'MEAN_SENT2',  'MEAN_SENT1_PCT', 'MEAN_SENT2_PCT', 'MEAN_SENT3_NB', 'MEAN_SENT3_NB_PCT',
                                'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
                                 'FOLLOWERS'])
print("# -- Logistic Regression: All Variables -- #")
print("Mean accuracy: ", log_out1[1])
print("Intercept: ", log_out1[3])
print("Coefficients : ", log_out1[4])
print("##########################################################")
print("##########################################################")
log_out2 = logistic_regression(data_train_log, data_test_log, price_train_log, price_test_log,
                               ['MEAN_SENT3_NB_PCT','MEAN_SENT3_NB','FOLLOWERS'])

##########################################
# 5. Random Forest Regression: Predict Stock price and Direction:
##########################################
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti

# -- 1. Predict Stock price
rf_model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_model.fit(data_train_lin, price_train_lin)
pred_rf  = rf_model.predict(data_test_lin)
acc_rf = rf_model.score(data_test_lin, price_test_lin)

print("# -- Random Forest Regression: All variables -- #")
print('Mean Squared Error:', mean_squared_error(price_test_lin, pred_rf))
print('Mean Absolute Error:', mean_absolute_error(price_test_lin, pred_rf))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(price_test_lin, pred_rf)))
print('R-Squared:', r2_score(price_test_lin, pred_rf))
print('Median Absolute Error:', median_absolute_error(price_test_lin, pred_rf))
print("##########################################################")
print("##########################################################")

print("# -- Random Forest: Important Variables -- #")
feature_imp = pd.Series(rf_model.feature_importances_, index= data_test_lin.columns).sort_values(ascending=False)
print(feature_imp)
print("##########################################################")
print("##########################################################")

print("# -- Random Forest: Contribution for Decision (egs - All variables) -- #")
predictions_egs = data_test_lin[0:2]
prediction, bias, contributions = ti.predict(rf_model, predictions_egs)
for i in range(len(predictions_egs)):
    print("Prediction", i)
    print( "Contribution by Top Feature:")
    for c, feature in sorted(zip(contributions[i], data_test_lin.columns))[0:2]:
        print(feature, round(c, 2))
    print( "-"*20)
print("##########################################################")
print("##########################################################")

# -- Using PCA:
rf_model_pc = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_model_pc.fit(df_pcs_train_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_train_lin)
pred_rf_pca  = rf_model_pc.predict(df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])
acc_rf_pca = rf_model_pc.score(df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_test_lin)
print("# -- Random Forest Regression: PCA 5 variables -- #")
print('Mean Squared Error:', mean_squared_error(price_test_lin, pred_rf_pca))
print('Mean Absolute Error:', mean_absolute_error(price_test_lin, pred_rf_pca))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(price_test_lin, pred_rf_pca)))
print('R-Squared:', r2_score(price_test_lin, pred_rf_pca))
print('Median Absolute Error:', median_absolute_error(price_test_lin, pred_rf_pca))
print("##########################################################")
print("##########################################################")

print("# -- Random Forest: Contribution for Decision (30 egs - PCA) -- #")
predictions_egs_pca = df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']][0:2]
prediction_pc, bias, contributions_pc = ti.predict(rf_model_pc, predictions_egs_pca)
for i in range(len(predictions_egs_pca)):
    print("Prediction", i)
    print( "Contribution by Top Feature:")
    for c, feature in sorted(zip(contributions_pc[i], df_pcs_test_lin[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].columns))[0:2]:
        print(feature, round(c, 2))
    print( "-"*20)
print("##########################################################")
print("##########################################################")

# -- 2. Predict Direction: Better than logistic?
rf_model_dir = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rf_model_dir.fit(df_pcs_train_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_train_log)
pred_rf_pca_dir  = rf_model_dir.predict(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])
acc_rf_pca_dir = rf_model_dir.score(df_pcs_test_log[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], price_test_log)
print("# -- Random Forest Classifier: PCA 5 Variables  -- #")
print("Mean accuracy: ", acc_rf_pca_dir)
print("##########################################################")
print("##########################################################")




