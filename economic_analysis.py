# Des:
# By:

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import missingno as msno
import pandas as pd
#warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import PCA
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import *
import functools
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.pyplot import ion
ion() # enables interactive mode

##########################################################################
# EXTRACT: Read in raw dataset as sourced from get_datasets.py
##########################################################################
all_data = pd.read_csv(r"C:\Users\btier\Documents\economic_data.csv")

##########################################################################
# TRANSFORM: Clean data and prepare for analysis
##########################################################################

###############################################
# 1. Check data: null values
###############################################
msno.matrix(all_data, figsize= (50,30))
# -- Reduce columns:
new_data = all_data.iloc[:, [0,1,2,3,4,7,8,10,11,12,13,15,16,17,27,28,30,40,42,43,44,45,46,47,48,49,50,51,54,55,56,57,
                             58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87]]

msno.matrix(new_data)
new_data_reduce1 = new_data[len(new_data['WMB_VOL'])-len(new_data['WMB_VOL'][new_data['WMB_VOL'].notna()]):len(new_data['WMB_VOL'])]

msno.matrix(new_data_reduce1)
new_data_reduce2 = new_data_reduce1.fillna(method='ffill')

msno.matrix(new_data_reduce2)

###############################################
# Impute values: Back fill: Appropriate for stock timeseries data
###############################################
new_data_reduce3 = new_data_reduce2

# -- Check data: clean dataset:
msno.matrix(new_data_reduce3)
sns.heatmap(new_data_reduce3.isnull(), cbar=False)
new_data_reduce3.to_csv(r"C:\Users\btier\Documents\new_data_reduce3.csv", index=False)

######################################################################################
# ANALYSIS:
# 1. Explore dataset.
# 2. Split dataset: Train, test, validation and all.
#    - Also included entire dataset, as PCA will be performed 4 times.
# 3. Linear Regression: Using All variables vs VIF results
# 4. PCA: Principal Component Analysis: PCA done on each dataset sepearately
#    - First regression run does PCA on train, test and validation sperately.
#    - Second time, PCA done once before splitting dataset (all data).
# 5. PCR: Principal Component Regression: PCA done on each dataset sepearately
# 6. PCA and PCR: Done on entire dataset first, then applied
######################################################################################

#########################################################
# 1. Explore Dataset:
#########################################################
sns.set(style="white")

###############################################
# Prepare data: Seperate dependent and independent variables
###############################################
gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_OPEN']
del new_data_reduce3['GSPC_LOW']
del new_data_reduce3['GSPC_ADJ_CLOSE']
del new_data_reduce3['GSPC_VOL']

###############################################
# Plot: Correlation Matrix Plot:
###############################################
corr_mx = new_data_reduce3.corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# -- temporarily remove date
date_tmp = new_data_reduce3['DATE']
del new_data_reduce3['DATE']

###############################################
# VIF: Variation Inflation Factor - Suggests which variables to keep.
###############################################

vif_df = pd.DataFrame()
vif_df["VIF_FACTOR"] = [variance_inflation_factor(new_data_reduce3.values, i) for i in range(new_data_reduce3.shape[1])]
vif_df["FEATURES"] = new_data_reduce3.columns
vif_df["VIF_FACTOR"] = [round(i,2) for i in vif_df["VIF_FACTOR"]]
predictor_variables = vif_df[vif_df["VIF_FACTOR"] < 10]
print("# -- VIF Factors < 10 -- #")
print(predictor_variables)
vif_factors = predictor_variables['FEATURES']
new_data_reduce3['DATE'] = date_tmp

#########################################################
# 2. Split datasets:
#########################################################

# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.96):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.96):]
# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.96)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.96)]
data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.2, random_state=0, shuffle=True)

val_date = validation_data['DATE']
train_date = data_train['DATE']
test_date = data_test['DATE']
del data_train['DATE']
del data_test['DATE']
del validation_data['DATE']

# -- All data:
all_data_pca = new_data_reduce3
all_gspc_pca = gspc_px

#########################################################
# 3. Linear Regression: All Variables v VIF
#########################################################

###############################################
# 1. Run OLS regression using ALL predictors:
###############################################
lr_model_all_vars = LinearRegression()
lr_model_all_vars.fit(data_train, gspc_px_train)
# -- Test OLS regression using ALL predictors:
predictions_test = lr_model_all_vars.predict(data_test)

# -- Find Metrics and Visualise:
print("# -- OLS Test Results: All Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_test))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_test)))
print('R-Squared:', r2_score(gspc_px_test, predictions_test))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_test))

###############################################
# 2. Run OLS regression using VIF predictors:
###############################################
lr_model_vif_vars = LinearRegression()
lr_model_vif_vars.fit(data_train[[i for i in vif_factors]], gspc_px_train)
prediction_vif = lr_model_vif_vars.predict(data_test[[i for i in vif_factors]])

# -- Find Metrics and Visualise:
print("# -- VIF Test Results: VIF Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, prediction_vif))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, prediction_vif))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, prediction_vif)))
print('R-Squared:', r2_score(gspc_px_test, prediction_vif))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, prediction_vif))

#########################################################
# 4. PCA: Principal Component Analysis: K-fold validation
#########################################################

###############################################
# 1. On training dataset:
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
lr_model_pca = LinearRegression()
# -- Use MSE as an indicator for closest fit:
mse_pca_2 = []
# -- Looping through X number of PC's, appends the MSE to above list. Will be used to find best model.
for i in np.arange(1, 50):
    # Multiple by -1 to negate the scoring method
    mse_result = -1 * model_selection.cross_val_score(lr_model_pca, data_reduced_train[:, :i], gspc_px_train.ravel(),
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

###############################################
# 2. On all dataset:
###############################################
del all_data_pca['DATE']

pca4 = PCA()
data_reduced_all = pca4.fit_transform(scale(all_data_pca))
kfold_cv_10_all = model_selection.KFold(n_splits=10, random_state=0, shuffle=True)
lr_model_pca_all = LinearRegression()
mse_pca_all = []
for i in np.arange(1, 50):
    # Multiple by -1 to negate the scoring method
    mse_result = -1 * model_selection.cross_val_score(lr_model_pca_all, data_reduced_all[:, :i], gspc_px.ravel(),
                                                      cv=kfold_cv_10_all,scoring='neg_mean_squared_error').mean()
    mse_pca_all.append(mse_result)

# -- Plot elbow graph of MSE
plt.figure()
plt.plot(mse_pca_all, '-v')
plt.xlabel('Principal Components in Linear Regression Model (all)')
plt.ylabel('MSE - Mean Squared Error')
plt.title('Elbow Chart - PCA K-Fold Cross Validation (all)')
plt.show()
# -- Plot elbow graph of variance
variance_explained_all = np.cumsum(np.round(pca4.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_all)
plt.xlabel('Principal Components in Linear Regression Model (all)')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component (all)')
plt.show()

#########################################################
# PCR: Principal Component Regression: Train and test (apply pca to both train and test)
#########################################################

# -- Initialise PCA class
pca3 = PCA()
data_reduced_train = pca3.fit_transform(scale(data_train))
data_reduced_test = pca3.fit_transform(scale(data_test))

# -- Initialise LR model
lr_model_pca = LinearRegression()

# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_pca.fit(data_reduced_train[:,:10], gspc_px_train)

# -- Run model:
predictions_2 = lr_model_pca.predict(data_reduced_test[:,:10])

# -- Find Metrics and Visualise:
print("# -- PCR Test Results: 10 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_2))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_2)))
print('R-Squared:', r2_score(gspc_px_test, predictions_2))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_2))

# -- Print Equation:
intercept = lr_model_pca.intercept_
coefs = lr_model_pca.coef_
print("# -- PCR Regression Equation -- #")
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
print("Regression Equation 10 PC's = ", round(intercept), '+ (', pc[0],round(coefs[0]), ') + (' ,pc[1],round(coefs[1]), ') + (', pc[2],round(coefs[2]),
      ') + (' ,pc[3],round(coefs[3]), ') + (', pc[4],round(coefs[4]), ') + (' ,pc[5],round(coefs[5]), ') + (' ,pc[6],round(coefs[6]), ') + (' ,pc[7],round(coefs[7]),
      ') + (' ,pc[8],round(coefs[8]), ') + (' ,pc[9],round(coefs[9]), ')')

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))

#########################################################
# PCR: Principal Component Regression: All data (do not apply pca - already done to all data)
#########################################################

# -- Extract validation subset: Keeping for last - never tested on
val_data = data_reduced_all[int(len(data_reduced_all)*0.96):]
val_gspc_px = gspc_px[int(len(gspc_px)*0.96):]

# -- Test / Train split:
non_val_data = data_reduced_all[:int(len(data_reduced_all)*0.96)]
non_val_gspc = gspc_px[:int(len(gspc_px)*0.96)]
data_train_all, data_test_all, gspc_px_train_all, gspc_px_test_all = train_test_split(non_val_data, non_val_gspc, test_size=0.2, random_state=0, shuffle=True)

lr_model_pca_all = LinearRegression()
lr_model_pca_all.fit(data_train_all[:,:10], gspc_px_train_all)
predall = lr_model_pca_all.predict(data_test_all[:,:10])

# -- Find Metrics and Visualise:
print("# -- PCR Test Results 2 (all data): 10 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predall))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predall))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predall)))
print('R-Squared:', r2_score(gspc_px_test, predall))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predall))

#########################################################
# Validation: Compare results of OLS on all variables vs PCR:
#########################################################

###############################################
# 1. Validate OLS regression using ALL predictors:
###############################################
val_all_pred = lr_model_all_vars.predict(validation_data)

# -- Find Metrics and Visualise:
print("# -- OLS Validation Results: All Variables -- #")
print('Mean Squared Error:', mean_squared_error(validation_gspc_px, val_all_pred))
print('Mean Absolute Error:',  mean_absolute_error(validation_gspc_px, val_all_pred))
print('Root Mean Squared Error:', np.sqrt( mean_squared_error(validation_gspc_px, val_all_pred)))
print('R-Squared:', r2_score(validation_gspc_px, val_all_pred))
print('Median Absolute Error:', median_absolute_error(validation_gspc_px, val_all_pred))

df_val_all_compare = pd.DataFrame({"DATE":val_date,'ACTUAL_PRICE': validation_gspc_px,'PREDICTED_PRICE': val_all_pred.flatten()})

print("# -- Comprare Validation Results: All Variables -- #")
print(df_val_all_compare.tail(10))
# -- Plot Predictions against Actual Prices:
plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_val_all_compare.index))], df_val_all_compare['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_val_all_compare.index))], df_val_all_compare['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))
plt.show()

###############################################
# 2. Validate OLS regression using PCR predictors: When PCA is done at each stage
###############################################

data_reduced_val = pca3.fit_transform(scale(validation_data))
val_pcr_pred = lr_model_pca.predict(data_reduced_val[:,:10])

# -- Find Metrics and Visualise:
print("# -- PCR Validation Results: 10 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(validation_gspc_px, val_pcr_pred))
print('Mean Absolute Error:',  mean_absolute_error(validation_gspc_px, val_pcr_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(validation_gspc_px, val_pcr_pred)))
print('R-Squared:', r2_score(validation_gspc_px, val_pcr_pred))
print('Median Absolute Error:', median_absolute_error(validation_gspc_px, val_pcr_pred))

df_val_pcr_compare = pd.DataFrame({"DATE":val_date, 'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': val_pcr_pred.flatten()})
print("# -- Comprare Validation Results: PCR Variables -- #")
print(df_val_pcr_compare.tail(10))
# -- Plot Predictions against Actual Prices:
plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_val_pcr_compare.index))], df_val_pcr_compare['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_val_pcr_compare.index))], df_val_pcr_compare['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))
plt.show()

###############################################
# 3. Validate OLS regression using PCR predictors: When PCA is done once
###############################################

pd = lr_model_pca_all.predict(val_data[:,:10])
print("# -- Comprare Validation Results: PCR Variables (PCA once / al data) -- #")
print('Mean Squared Error:', mean_squared_error(val_gspc_px, pd))
print('Mean Absolute Error:', mean_absolute_error(val_gspc_px, pd))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(val_gspc_px, pd)))
print('R-Squared:', r2_score(val_gspc_px, pd))
print('Median Absolute Error:', median_absolute_error(val_gspc_px, pd))
