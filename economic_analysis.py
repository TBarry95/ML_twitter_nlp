# Des:
# By:

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
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

######################################################################################
# Extract: Read in raw dataset as sourced from get_datasets.py
######################################################################################
all_data = pd.read_csv(r"C:\Users\btier\Documents\economic_data.csv")

######################################################################################
# Transform: Clean data and prepare for analysis
######################################################################################

# 1. Check data: null values
msno.matrix(all_data, figsize= (50,30))
# -- Reduce columns:
new_data = all_data.iloc[:, [0,1,2,3,4,7,8,10,11,12,13,15,16,17,27,28,30,40,42,43,44,45,46,47,48,49,50,51,54,55,56,57,
                             58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87]]

# 2. Check data: null values
msno.matrix(new_data)
# -- reduce dataset again
new_data_reduce1 = new_data[len(new_data['WMB_VOL'])-len(new_data['WMB_VOL'][new_data['WMB_VOL'].notna()]):len(new_data['WMB_VOL'])]

# 3. Check data: null values
msno.matrix(new_data_reduce1)
new_data_reduce2 = new_data_reduce1.fillna(method='ffill')

# 4. Check data: null values
msno.matrix(new_data_reduce2)

# Impute values: Back fill: - CONF_INX_INST, BUILD_COST_INX, US_POP_MILL
new_data_reduce3 = new_data_reduce2

# 5. Check data: clean dataset:
msno.matrix(new_data_reduce3)
sns.heatmap(new_data_reduce3.isnull(), cbar=False)
new_data_reduce3.to_csv(r"C:\Users\btier\Documents\new_data_reduce3.csv", index=False)

######################################################################################
# Analysis: Regression
# 1. Explore dataset: Check for Multicollinearity + Plots.
# 2. Linear Regression: OLS: Using All variables vs VIF results + Plots.
# 3. PCA: Principal Component Analysis + Plots.
# 4. PCR: Principal Component Regression + Plots.
######################################################################################

#########################################################
# # 1. Explore Dataset: Check for Multicollinearity
#########################################################
sns.set(style="white")

# -- Prepare data: Seperate dependent and independent variables
gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_OPEN']
del new_data_reduce3['GSPC_LOW']
del new_data_reduce3['GSPC_ADJ_CLOSE']
del new_data_reduce3['GSPC_VOL']

# -- Plot: Correlation Matrix Plot:
corr_mx = new_data_reduce3.corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# -- temporarily remove date
date_tmp = new_data_reduce3['DATE']
del new_data_reduce3['DATE']

# -- VIF: Variation Inflation Factor - Suggests which variables to keep.
vif_df = pd.DataFrame()
vif_df["VIF_FACTOR"] = [variance_inflation_factor(new_data_reduce3.values, i) for i in range(new_data_reduce3.shape[1])]
vif_df["FEATURES"] = new_data_reduce3.columns
vif_df["VIF_FACTOR"] = [round(i,2) for i in vif_df["VIF_FACTOR"]]
print(vif_df)
predictor_variables = vif_df[vif_df["VIF_FACTOR"] < 10]
print(predictor_variables)
vif_factors = predictor_variables['FEATURES']

new_data_reduce3['DATE'] = date_tmp

#########################################################
# Split datasets:
#########################################################
'''
# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.96):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.96):]

# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.96)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.96)]

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0)'''

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.3, random_state=0)

del data_train['DATE']
del data_test['DATE']
# del validation_data['DATE']
'''del gspc_px_train['index']
del gspc_px_test['index']
del data_train['index']
del data_test['index']'''

#########################################################
# Linear Regression: VIF results vs All Variables
#########################################################

# -- Run OLS regression using ALL predictors:
lr_model_all_train = sm.OLS(gspc_px_train, data_train).fit()
predictions_test = lr_model_all_train.predict(data_test)
model_summary_train = lr_model_all_train.summary()
print(model_summary_train)

# -- Validate OLS regression using ALL predictors:
predictions_all = lr_model_all_train.predict(validation_gspc_px)

# -- Find Metrics and Visualise:
mse_pred_all_vars = mean_squared_error(validation_gspc_px, predictions_all)
mae_pred_all_vars = mean_absolute_error(validation_gspc_px, predictions_all)
rmse_pred_all_vars = np.sqrt(mse_pred_all_vars)
r2_all_vars = r2_score(validation_gspc_px, predictions_all)
median_ae_all_vars = median_absolute_error(validation_gspc_px, predictions_all)
print('Mean Squared Error:', mse_pred_all_vars)
print('Mean Absolute Error:', mae_pred_all_vars)
print('Root Mean Squared Error:', rmse_pred_all_vars)
print('R-Squared:', r2_all_vars)
print('Median Absolute Error:', median_ae_all_vars)

compare_validation = pd.DataFrame({'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': predictions_all})
plt.figure()
plot1, = plt.plot([i for i in range(0,len(compare_validation.index))], compare_validation['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(compare_validation.index))], compare_validation['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))

'''
# -- Compare results in table format:
df_compare_validation = pd.DataFrame({'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': predictions_pcr.flatten()})
print(df_compare_validation.head(30))

# -- Plot Predictions against Actual Prices: 
plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))'''
'''
# -- Test OLS regression using VIF predictors:
lr_model_vif_train = sm.OLS(gspc_px_train, data_train[[i for i in vif_factors]]).fit()
predictions_vif_test = lr_model_vif_train.predict(data_test[[i for i in vif_factors]])
model_summary_vif_train = lr_model_vif_train.summary()
print(model_summary_vif_train)

# -- Validate OLS regression using VIF predictors:
predictions_vif = lr_model_vif_train.predict(validation_data[[i for i in vif_factors]])
model_summary_vif = lr_model_vif_train.summary()
print(model_summary_vif)

# -- Find Metrics and Visualise:
mse_pred_vif = mean_squared_error(validation_gspc_px, predictions_vif)
mae_pred_vif = mean_absolute_error(validation_gspc_px, predictions_vif)
rmse_pred_vif = np.sqrt(mse_pred_vif)
r2_vif = r2_score(validation_gspc_px, predictions_vif)
median_ae_vif = median_absolute_error(validation_gspc_px, predictions_vif)
print('Mean Squared Error:', mse_pred_vif)
print('Mean Absolute Error:', mae_pred_vif)
print('Root Mean Squared Error:', rmse_pred_vif)
print('R-Squared:', r2_vif)
print('Median Absolute Error:', median_ae_vif)
'''
#########################################################
# PCA: Principal Component Analysis: On training data
#########################################################

# -- Set date as index, so that it can easily be reapplied once removed
new_data_reduce3 = new_data_reduce3.set_index(new_data_reduce3['DATE'])
del new_data_reduce3['DATE']
#del new_data_reduce3['GSPC_VOL']

# -- Initialise PCA class
pca2 = PCA()

# -- Standardise/scale the training data such that each column's mean = 0
data_reduced_train = pca2.fit_transform(scale(data_train))
print(pd.DataFrame(pca2.components_.T).head())

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

# -- Plot elbow graph of variance
variance_explained_2 = np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
plt.figure()
plt.plot(variance_explained_2)
plt.xlabel('Principal Components in Linear Regression Model')
plt.ylabel('% Variance Explained')
plt.title('Elbow Chart - Variance Explained by Principal Component')

#########################################################
# PCR: Principal Component Regression: Train and test LR model based on PCA.
#########################################################

# -- Standardise/scale the test data such that each column's mean = 0
data_reduced_test = pca2.fit_transform(scale(data_test))
print(pd.DataFrame(pca2.components_.T).head())

# -- Initialise LR model
lr_model_pca_run = LinearRegression()

# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_pca_run.fit(data_reduced_train[:,:10], gspc_px_train)

# -- Run model:
predictions_2 = lr_model_pca_run.predict(data_reduced_test[:,:10])

# -- Find Metrics and Visualise:
mse_pred_2 = mean_squared_error(gspc_px_test, predictions_2)
mae_pred_2 = mean_absolute_error(gspc_px_test, predictions_2)
rmse_pred_2 = np.sqrt(mse_pred_2)
r2 = r2_score(gspc_px_test, predictions_2)
median_ae = median_absolute_error(gspc_px_test, predictions_2)
print('Mean Squared Error:', mse_pred_2)
print('Mean Absolute Error:', mae_pred_2)
print('Root Mean Squared Error:', rmse_pred_2)
print('R-Squared:', r2)
print('Median Absolute Error:', median_ae)

# -- Print Equation:
intercept = lr_model_pca_run.intercept_
coefs = lr_model_pca_run.coef_
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
print("Regression Equation 6 PC's: ", round(intercept), '+ (', pc[0],round(coefs[0]), ') + (' ,pc[1],round(coefs[1]), ') + (', pc[2],round(coefs[2]),
      ') + (' ,pc[3],round(coefs[3]), ') + (', pc[4],round(coefs[4]), ') + (' ,pc[5],round(coefs[5]), ') + (' ,pc[6],round(coefs[6]), ') + (' ,pc[7],round(coefs[7]),
      ') + (' ,pc[8],round(coefs[8]), ') + (' ,pc[9],round(coefs[9]), ')')

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
print(df_compare.head(30))


'''
del validation_data['DATE']
# -- Run PCR on Validation dataset:
data_reduced_validation = pca2.fit_transform(scale(validation_data))
print(pd.DataFrame(pca2.components_.T).head())

# -- Run model:
predictions_pcr = lr_model_pca_run.predict(data_reduced_validation[:,:10])

# -- Find Metrics and Visualise:
mse_pred_pcr = mean_squared_error(validation_gspc_px, predictions_pcr)
mae_pred_pcr = mean_absolute_error(validation_gspc_px, predictions_pcr)
rmse_pred_pcr = np.sqrt(mse_pred_pcr)
r2_pcr = r2_score(validation_gspc_px, predictions_pcr)
median_ae_pcr = median_absolute_error(validation_gspc_px, predictions_pcr)

print('Mean Squared Error:', mse_pred_pcr)
print('Mean Absolute Error:', mae_pred_pcr)
print('Root Mean Squared Error:', rmse_pred_pcr)
print('R-Squared:', r2_pcr)
print('Median Absolute Error:', median_ae_pcr)

# -- Compare results in table format:
df_compare_validation = pd.DataFrame({'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': predictions_pcr.flatten()})
print(df_compare_validation.head(30))

plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))'''
