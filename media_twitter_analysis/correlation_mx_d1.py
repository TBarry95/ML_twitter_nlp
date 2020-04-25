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
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pyplot import ion
ion() # enables interactive mode

##########################################################################
# A. EXTRACT: Read in cleaned dataset
##########################################################################
df_features = pd.read_csv(r".\media_data_cleaned.csv")


##########################################
# Correlation Matrix: Tweet Sentiment and Stock price (and more)
##########################################

# -- Plot: Correlation Matrix Plot:
corr_mx = df_features[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL', 'PCT_SENT_NB',
       'FAV_COUNT_DAY', 'RT_COUNT_DAY', 'TWEET_COUNT_DAY', 'LEN_TWEET_SUM',
       'FOLLOWERS', 'Close']].corr()

mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print("# -- Results: Correlation Matrix: -- #")
print(corr_mx)
print("No significant correlation between SP500 close and any tweet sentiment metrics (although positive")
print("##########################################################")
print("##########################################################")