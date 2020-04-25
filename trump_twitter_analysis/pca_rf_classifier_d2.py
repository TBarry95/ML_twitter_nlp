# Des: Analysis script of Donald Trumps tweets in order to predict stock price direction
#      using sentiment analysis, correlation matrix, and logistic regression.
# By: Tiernan Barry - x19141840 (NCI) - Data Mining and Machine Learning

# Libraries:
import Twitter_API_Module as twt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
# Source files (functions):
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


##########################################################################
# EXTRACT:
##########################################################################
all_data = pd.read_csv(r".\trump_data_cleaned.csv")

##########################################
# 2. Split data:
##########################################

ind_vars = all_data[[ 'MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT',  'FAV_COUNT', 'RT_COUNT']]

dep_var1 = all_data['SP_CLOSE']
dep_var2 = all_data['SP_DIRECTION']

# --  Random Forest
data_train_rf, data_test_rf, price_train_rf, price_test_rf = train_test_split(ind_vars,dep_var1 , test_size=0.2, random_state=0, shuffle=True)
# -- Logistic Regression
data_train_log, data_test_log, price_train_log, price_test_log = train_test_split(ind_vars, dep_var2, test_size=0.2, random_state=0, shuffle=True)

##########################################
# 3. PCA
##########################################


# -- Logistic Regression
fts_for_pca_log = data_train_rf[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT']]
fts_for_pca_test_log = data_test_rf[['MEAN_SENT_LEX_CATG', 'MEAN_SENT_LEX_POL', 'MEAN_SENT_NB',
       'PCT_SENT_NB', 'PCT_SENT_LEX_CATG', 'PCT_SENT_LEX_POL',
       'DIRECTION_LEX_CATG', 'DIRECTION_LEX_POL', 'DIRECTION_NB','TWEET_COUNT']]

# -- Initialise PCA class
pca2 = PCA()
data_reduced_train_log = pca2.fit_transform(scale(fts_for_pca_log))
data_reduced_test_log = pca2.transform(scale(fts_for_pca_test_log))

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')

print("# -- Test Results - PCA: Variance Explained per PC -- #")
print(variance_explained_2)
print("##########################################################")

##########################################
#  PCR: RF Classifier: Predict Stock price Direction
##########################################

acc = []
for i in range(1, 11):
    rf_class = RandomForestClassifier()
    rf_class.fit(data_reduced_train_log[:, :i], price_train_log)
    pred = rf_class.predict(data_reduced_test_log[:, :i])  # predcition
    accuracy = rf_class.score(data_reduced_test_log[:, :i], price_test_log)  # Return the mean accuracy on the given test data and labels.
    acc.append([i, accuracy ])

acc_list = [i[1] for i in acc]
plt.figure()
plt.plot(acc_list)
plt.xlabel("Number of Principal Components")
plt.ylabel("Mean Accuracy")
plt.title("All PC Logistic Regression Models - Mean Accuracy")

print("# -- Test Results: Optimal PC RF Classification -- #")
print("Mean Accuracy: ", sorted(acc_list)[-1])
print("##########################################################")
print("##########################################################")






