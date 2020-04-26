# Des: Using multiple quantitative and financial data points, this script conducts
#      regression analysis, PCA and and PCR.
# By: Tiernan Barry - x19141840 - NCI

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from sklearn.decomposition import PCA
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from sklearn.metrics import *
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.pyplot import ion
ion() # enables interactive mode

##########################################################################
#  EXTRACT: Read in cleaned dataset
##########################################################################
new_data_reduce3 = pd.read_csv(r".\quant_data_cleaned.csv")


###############################################
# Prepare data: Seperate dependent and independent variables
###############################################

gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_OPEN']
del new_data_reduce3['GSPC_LOW']
del new_data_reduce3['GSPC_ADJ_CLOSE']
del new_data_reduce3['GSPC_VOL']
del new_data_reduce3['GSPC_HIGH']

#########################################################
# Split datasets:
#########################################################
'''
# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.85):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.85):]
# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.85)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.85)]'''

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.3, random_state=0, shuffle=True)

'''val_date = validation_data['DATE']
del validation_data['DATE']'''
train_date = data_train['DATE']
test_date = data_test['DATE']
del data_train['DATE']
del data_test['DATE']

#########################################################
# PCA: Principal Component Analysis: K-fold validation
#########################################################

###############################################
# On training dataset:
###############################################
# -- Set date as index, so that it can easily be reapplied once removed
new_data_reduce3 = new_data_reduce3.set_index(new_data_reduce3['DATE'])
del new_data_reduce3['DATE']

# -- Initialise PCA class
pca2 = PCA()
# -- Standardise/scale the training data such that each column's mean = 0
data_reduced_train = pca2.fit_transform(scale(data_train))
# -- Find optimal number of components by applying k-fold Cross Validation
kfold_cv_10_2 = model_selection.KFold(n_splits=10, random_state=0, shuffle=True)
# -- Initialise LR model
ridge_model_pca = Ridge()
# -- Use MSE as an indicator for closest fit:
mse_pca_2 = []
# -- Looping through X number of PC's, appends the MSE to above list. Will be used to find best model.
for i in np.arange(1, 50):
    # Multiple by -1 to negate the scoring method
    mse_result = -1 * model_selection.cross_val_score(ridge_model_pca, data_reduced_train[:, :i], gspc_px_train.ravel(),
                                                      cv=kfold_cv_10_2,scoring='neg_mean_squared_error').mean()
    mse_pca_2.append(mse_result)
# -- Plot elbow graph of MSE
plt.figure()
plt.plot(mse_pca_2, '-v')
plt.xlabel('Principal Components in Linear Regression Model')
plt.ylabel('MSE - Mean Squared Error')
plt.title('Elbow Chart - PCA K-Fold Cross Validation (Training)')
plt.show()

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Linear Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')
plt.show()

print("Optimal Number of Principal Components between 10-15 based on elbow graph")
print("##########################################################")
print("##########################################################")

#########################################################
# PCR: Principal Component Ridge Regression: Train and test (apply pca to both train and test)
#########################################################

for i in [1,2,3,4,5,6,7,8,10,12,14,16]:
    # -- Initialise PCA class
    pca3 = PCA()
    data_reduced_train = pca3.fit_transform(scale(data_train))
    data_reduced_test = pca3.fit_transform(scale(data_test))
    #v = pca3.fit_transform(scale(validation_data))
    # -- Initialise Ridge model
    ridge_pca = Ridge()
    # -- Fit Ridge model: based on Elbow graph
    ridge_pca.fit(data_reduced_train[:,:i], gspc_px_train)
    # -- Run model:
    predictions_2 = ridge_pca.predict(data_reduced_test[:,:i])

    # -- Find Metrics and Visualise:
    print("# -- Test Results - PCR: ", i, " PCA Variables -- #")
    print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_2))
    print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_2))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_2)))
    print('R-Squared:', r2_score(gspc_px_test, predictions_2))
    print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_2))
    print("##########################################################")
    print("##########################################################")

'''params = {'alpha': [0.001, 0.01, 0.03, 0.05, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.4, 0.6, 0.8, 1, 5, 15, 20, 25, 30]}
ridge_reg = GridSearchCV(ridge_cv, params, scoring='neg_mean_squared_error', cv=20)
ridge_reg.fit(data_train, gspc_px_train)'''


from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.15, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 5.5,6, 7, 8, 9, 10 , 11 ,15, 20, 25, 30]}
ridge_pcacv = Ridge()

ridge_reg = GridSearchCV(ridge_pcacv, params, scoring='neg_mean_squared_error', cv=20)
ridge_reg.fit(data_reduced_train[:,:5], gspc_px_train)

print(ridge_reg.best_score_)
print(ridge_reg.best_params_)

ridge_pcacv = Ridge(alpha=6, random_state=1)
ridge_pcacv.fit(data_reduced_train[:,:5], gspc_px_train)

from yellowbrick.regressor import ResidualsPlot
plt.figure()
visualizer = ResidualsPlot(ridge_pcacv)
visualizer.fit(data_reduced_train[:,:5], gspc_px_train)  # Fit the training data to the visualizer
visualizer.score(data_reduced_test[:,:5], gspc_px_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))


'''
###############################################
# 2. Validate Ridge regression using PCR predictors: When PCA is done at each stage
###############################################

pca_pcr = PCA()

data_reduced_val = pca_pcr.fit_transform(scale(validation_data))
val_pcr_pred = ridge_pca.predict(data_reduced_val[:,:15])

# -- Find Metrics and Visualise:
print("# -- Validation Results - PCR: 10 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(validation_gspc_px, val_pcr_pred))
print('Mean Absolute Error:',  mean_absolute_error(validation_gspc_px, val_pcr_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(validation_gspc_px, val_pcr_pred)))
print('R-Squared:', r2_score(validation_gspc_px, val_pcr_pred))
print('Median Absolute Error:', median_absolute_error(validation_gspc_px, val_pcr_pred))
print("##########################################################")
print("##########################################################")

df_val_pcr_compare = pd.DataFrame({"DATE":val_date, 'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': val_pcr_pred.flatten()})
print("# -- Validation Results - Comprare: PCR Variables -- #")
print(df_val_pcr_compare.tail(10))
print("##########################################################")
print("##########################################################")
# -- Plot Predictions against Actual Prices:
plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_val_pcr_compare.index))], df_val_pcr_compare['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_val_pcr_compare.index))], df_val_pcr_compare['PREDICTED_PRICE'])
plt.xlabel('Number of days before April 16th 2020')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))
plt.show()

print("##########################################################")
print("##########################################################")



'''

