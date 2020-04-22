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
from sklearn.preprocessing import scale
from sklearn.metrics import *
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.pyplot import ion
ion() # enables interactive mode

##########################################################################
# A. EXTRACT: Read in cleaned dataset
##########################################################################
all_data = pd.read_csv(r".\quant_data_cleaned.csv")

new_data_reduce3 = all_data

######################################################################################
# C. ANALYSIS:
# 1. Explore dataset.
# 2. Split dataset: Train, test, validation and all.
#    - Also included entire dataset, as PCA will be performed 4 times.
# 3. Linear Regression: Using All variables vs VIF results
# 4. PCA: Principal Component Analysis: PCA done on each dataset sepearately
# 5. PCR: Principal Component Regression: PCA done on each dataset sepearately
# 6. PCA and PCR:
# 7. Cross validation LR models
######################################################################################

#########################################################
# 1. Explore Dataset for Regression:
#########################################################

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

###############################################
# Plot: Correlation Matrix Plot:
###############################################
sns.set(style="white")

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
print("##########################################################")
print("##########################################################")

#########################################################
# 2. Split datasets:
#########################################################

# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.85):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.85):]
# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.85)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.85)]

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0, shuffle=True)
val_date = validation_data['DATE']
del validation_data['DATE']
'''data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.2, random_state=0, shuffle=True)
'''
train_date = data_train['DATE']
test_date = data_test['DATE']
del data_train['DATE']
del data_test['DATE']

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
print("# -- Test Results - OLS: All", len(data_train.columns), "Variables  -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_test))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_test)))
print('R-Squared:', r2_score(gspc_px_test, predictions_test))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_test))
print("##########################################################")
print("##########################################################")

###############################################
# 2. Run OLS regression using VIF predictors:
###############################################
lr_model_vif_vars = LinearRegression()
lr_model_vif_vars.fit(data_train[[i for i in vif_factors]], gspc_px_train)
prediction_vif = lr_model_vif_vars.predict(data_test[[i for i in vif_factors]])

print("# -- Strong results but too many variables, need to reduce dimensions:  -- #")
print("# -- 1. Using VIF variables  -- #")
print("# -- 2. Using PCA variables  -- #")
print("##########################################################")
print("##########################################################")

# -- Find Metrics and Visualise:
print("# -- Test Results - OLS: VIF Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, prediction_vif))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, prediction_vif))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, prediction_vif)))
print('R-Squared:', r2_score(gspc_px_test, prediction_vif))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, prediction_vif))
print("##########################################################")
print("##########################################################")

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


#########################################################
# PCR: Principal Component Regression: Train and test (apply pca to both train and test)
#########################################################

# -- Initialise PCA class
pca3 = PCA()
data_reduced_train = pca3.fit_transform(scale(data_train))
data_reduced_test = pca3.fit_transform(scale(data_test))
v = pca3.fit_transform(scale(validation_data))
# -- Initialise LR model
lr_model_pca = LinearRegression()
# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_pca.fit(data_reduced_train[:,:20], gspc_px_train)
# -- Run model:
predictions_2 = lr_model_pca.predict(data_reduced_test[:,:20])

# -- Find Metrics and Visualise:
print("# -- Test Results - PCR: 10 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_2))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_2))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_2)))
print('R-Squared:', r2_score(gspc_px_test, predictions_2))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_2))
print("##########################################################")
print("##########################################################")

# -- Print Equation:
intercept = lr_model_pca.intercept_
coefs = lr_model_pca.coef_
print("# -- PCR Regression Equation -- #")
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
print("Regression Equation 10 PC's = ", round(intercept), '+ (', pc[0],round(coefs[0]), ') + (' ,pc[1],round(coefs[1]), ') + (', pc[2],round(coefs[2]),
      ') + (' ,pc[3],round(coefs[3]), ') + (', pc[4],round(coefs[4]), ') + (' ,pc[5],round(coefs[5]), ') + (' ,pc[6],round(coefs[6]), ') + (' ,pc[7],round(coefs[7]),
      ') + (' ,pc[8],round(coefs[8]), ') + (' ,pc[9],round(coefs[9]), ')')
print("##########################################################")
print("##########################################################")
# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))

# -- Initialise PCA class
pca_1pc = PCA()
data_reduced_train_1pc = pca_1pc.fit_transform(scale(data_train))
data_reduced_test_1pc = pca_1pc.fit_transform(scale(data_test))
# -- Initialise LR model
lr_model_pca_1pc = LinearRegression()
# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_pca_1pc.fit(data_reduced_train[:,:1], gspc_px_train)
# -- Run model:
predictions_2_1pc = lr_model_pca_1pc.predict(data_reduced_test[:,:1])

# -- Find Metrics and Visualise:
print("# -- Test Results - PCR: 1 PCA Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_2_1pc))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_2_1pc))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_2_1pc)))
print('R-Squared:', r2_score(gspc_px_test, predictions_2_1pc))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_2_1pc))
print("##########################################################")
print("##########################################################")

######################################################################################
# D. RESULTS: Validation: Compare results of OLS on all variables vs PCR:
######################################################################################

###############################################
# 1. Validate OLS regression using ALL predictors:
###############################################
val_all_pred = lr_model_all_vars.predict(validation_data)

# -- Find Metrics and Visualise:
print("# -- Validation Results - OLS: All Variables -- #")
print('Mean Squared Error:', mean_squared_error(validation_gspc_px, val_all_pred))
print('Mean Absolute Error:',  mean_absolute_error(validation_gspc_px, val_all_pred))
print('Root Mean Squared Error:', np.sqrt( mean_squared_error(validation_gspc_px, val_all_pred)))
print('R-Squared:', r2_score(validation_gspc_px, val_all_pred))
print('Median Absolute Error:', median_absolute_error(validation_gspc_px, val_all_pred))
print("##########################################################")
print("##########################################################")
df_val_all_compare = pd.DataFrame({"DATE":val_date,'ACTUAL_PRICE': validation_gspc_px,'PREDICTED_PRICE': val_all_pred.flatten()})

print("# -- Validation Results - Comprare: All Variables -- #")
print(df_val_all_compare.tail(10))
print("##########################################################")
print("##########################################################")
# -- Plot Predictions against Actual Prices:
plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_val_all_compare.index))], df_val_all_compare['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_val_all_compare.index))], df_val_all_compare['PREDICTED_PRICE'])
plt.xlabel('Number of days before April 16th 2020')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))
plt.show()

###############################################
# 2. Validate OLS regression using PCR predictors: When PCA is done at each stage
###############################################

pca_pcr = PCA()

data_reduced_val = pca_pcr.fit_transform(scale(validation_data))
val_pcr_pred = lr_model_pca.predict(data_reduced_val[:,:20])

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

###############################################
# Cross validation K-Fold regression:
###############################################
# Necessary imports:
from sklearn.model_selection import cross_val_score, cross_val_predict

linear_mdl_cv = LinearRegression()
# Perform 6-fold cross validation
lr_kf_r2 = []
for i in range(2,20):
    pred = cross_val_predict(linear_mdl_cv, new_data_reduce3, gspc_px, cv=i)
    lr_kf_r2.append(r2_score(gspc_px, pred))

pca_new = PCA()
reducer_all = pca_new.fit_transform(scale(new_data_reduce3))
lin_md = LinearRegression()

lr_kf_r2_pca = []
for i in range(2,20):
    pred = cross_val_predict(lin_md, reducer_all[:,20], gspc_px, cv=i)
    lr_kf_r2_pca.append(r2_score(gspc_px, pred))


