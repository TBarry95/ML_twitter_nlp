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

######################################################################################
# Extract: Read in raw dataset as sourced from get_datasets.py
######################################################################################
all_data = pd.read_csv(r"C:\Users\btier\Documents\economic_data.csv")

######################################################################################
# Transform: Clean data and prepare for analysis
######################################################################################

# 1. Check data: null values
msno.matrix(all_data, figsize= (50,30))

# Reduce columns - eg. 1 gold, 1 silver, etc..
new_data = pd.DataFrame({"DATE": all_data['DATE'],
                         "GOLD_USD_AM": all_data['GOLD_USD_AM'],
                         "SILVER_PRICE_USD": all_data['SILVER_PRICE_USD'],
                         "PLAT_OPEN_USD": all_data['PLAT_OPEN_USD'],
                         "FED_FUND_RATE": all_data['FED_FUND_RATE'],
                         "USD_GBP": all_data['USD_GBP'],
                         "BUILD_COST_INX": all_data['BUILD_COST_INX'],
                         "US_POP_MILL": all_data["US_POP_MILL"],
                         "CONF_INX_INST": all_data["CONF_INX_INST"],
                         "CONF_INX_INDV": all_data["CONF_INX_INDV"],
                         "HOUSE_PX_INX_REAL": all_data["HOUSE_PX_INX_REAL"],
                         "GSPC_VOL": all_data['GSPC_VOL'],
                         "GSPC_CLOSE": all_data['GSPC_CLOSE']})

# 2. Check data: null values
msno.matrix(new_data)

# reduce dataset to 11420 - so USDGBP is available
new_data_reduce1 = new_data[len(new_data['USD_GBP'])-len(new_data['USD_GBP'][new_data['USD_GBP'].notna()]):len(new_data['USD_GBP'])]

# 3. Check data: null values
msno.matrix(new_data_reduce1)

# Impute values: Forward fill: - specify which cols need this
new_data_reduce2 = pd.DataFrame({"DATE": new_data_reduce1['DATE'],
                         "GOLD_USD_AM": new_data_reduce1['GOLD_USD_AM'].fillna(method='ffill'),
                         "SILVER_PRICE_USD": new_data_reduce1['SILVER_PRICE_USD'].fillna(method='ffill'),
                         "PLAT_OPEN_USD": new_data_reduce1['PLAT_OPEN_USD'].fillna(method='ffill'),
                         "FED_FUND_RATE": new_data_reduce1['FED_FUND_RATE'].fillna(method='ffill'),
                         "USD_GBP": new_data_reduce1['USD_GBP'].fillna(method='ffill'),
                         "BUILD_COST_INX": new_data_reduce1['BUILD_COST_INX'].fillna(method='ffill'),
                         "US_POP_MILL": new_data_reduce1["US_POP_MILL"].fillna(method='ffill'),
                         "CONF_INX_INST": new_data_reduce1["CONF_INX_INST"].fillna(method='ffill'),
                         "CONF_INX_INDV": new_data_reduce1["CONF_INX_INDV"].fillna(method='ffill'),
                         "HOUSE_PX_INX_REAL": new_data_reduce1["HOUSE_PX_INX_REAL"].fillna(method='ffill'),
                         "GSPC_VOL": new_data_reduce1['GSPC_VOL'].fillna(method='ffill'),
                         "GSPC_CLOSE": new_data_reduce1['GSPC_CLOSE'].fillna(method='ffill')})

# 4. Check data: null values
msno.matrix(new_data_reduce2)

# Impute values: Back fill: - CONF_INX_INST, BUILD_COST_INX, US_POP_MILL
new_data_reduce3 = pd.DataFrame({"DATE": new_data_reduce2['DATE'],
                         "GOLD_USD_AM": new_data_reduce2['GOLD_USD_AM'],
                         "SILVER_PRICE_USD": new_data_reduce2['SILVER_PRICE_USD'],
                         "PLAT_OPEN_USD": new_data_reduce2['PLAT_OPEN_USD'],
                         "FED_FUND_RATE": new_data_reduce2['FED_FUND_RATE'],
                         "USD_GBP": new_data_reduce2['USD_GBP'],
                         "BUILD_COST_INX": new_data_reduce2['BUILD_COST_INX'].fillna(method='bfill'),
                         "US_POP_MILL": new_data_reduce2["US_POP_MILL"].fillna(method='bfill'),
                         "CONF_INX_INST": new_data_reduce2["CONF_INX_INST"].fillna(method='bfill'),
                         "CONF_INX_INDV": new_data_reduce2["CONF_INX_INDV"].fillna(method='bfill'),
                         "HOUSE_PX_INX_REAL": new_data_reduce2["HOUSE_PX_INX_REAL"].fillna(method='bfill'),
                         "GSPC_VOL": new_data_reduce2['GSPC_VOL'],
                         "GSPC_CLOSE": new_data_reduce2['GSPC_CLOSE']})

# 5. Check data: clean dataset:
msno.matrix(new_data_reduce3)
sns.heatmap(new_data_reduce3.isnull(), cbar=False)

new_data_reduce3.to_csv(r"C:\Users\btier\Documents\new_data_reduce3.csv", index=False)

######################################################################################
# Analysis: Regression - 1. PCA, 2. PCR, 3. Best Subsets Regression
######################################################################################

#########################################################
# 1. PCA: Principal Component Analysis: On training data
#########################################################

# -- Seperate dependent and independent variables
gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']

# -- Set date as index, so that it can easily be reapplied once removed
new_data_reduce3 = new_data_reduce3.set_index(new_data_reduce3['DATE'])
del new_data_reduce3['DATE']
#del new_data_reduce3['GSPC_VOL']

# -- Initialise PCA class
pca2 = PCA()

# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.99):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.99):]

# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.99)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.99)]
data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0)

# -- Standardise/scale the training data such that each column's mean = 0
data_reduced_train = pca2.fit_transform(scale(data_train))
print(pd.DataFrame(pca2.components_.T).head())

# -- Find optimal number of components by applying k-fold Cross Validation
kfold_cv_10_2 = model_selection.KFold(n_splits=10, random_state=0, shuffle=True)

# -- Initialise LR model
lr_model_2 = LinearRegression()

# -- Use MSE as an indicator for closest fit:
mse_pca_2 = []

# -- Looping through X number of PC's, appends the MSE to above list. Will be used to find best model.
for i in np.arange(1, 11):
    # Multiple by -1 to negate the scoring method
    mse_result = -1 * model_selection.cross_val_score(lr_model_2, data_reduced_train[:, :i], gspc_px_train.ravel(),
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
# 2. PCR: Principal Component Regression: Train and test LR model based on PCA.
#########################################################

# -- Standardise/scale the test data such that each column's mean = 0
data_reduced_test = pca2.fit_transform(scale(data_test))
print(pd.DataFrame(pca2.components_.T).head())

# -- Initialise LR model
lr_model_run_2 = LinearRegression()

# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_run_2.fit(data_reduced_train[:,:6], gspc_px_train)

# -- Run model:
predictions_2 = lr_model_run_2.predict(data_reduced_test[:,:6])

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

metrics = pd.DataFrame([mse_pred_2, mae_pred_2, rmse_pred_2])
metrics.plot(kind = 'bar')

# -- Print Equation:
intercept = lr_model_run_2.intercept_
coefs = lr_model_run_2.coef_
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
print("Regression Equation 6 PC's: ", round(intercept), '+ (', pc[0],round(coefs[0]), ') + (' ,pc[1],round(coefs[1]), ') + (', pc[2],round(coefs[2]),
      ') + (' ,pc[3],round(coefs[3]), ') + (', pc[4],round(coefs[4]), ') + (' ,pc[5],round(coefs[5]), ')')

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
print(df_compare.head(30))

# -- Run PCR on Validation dataset:
data_reduced_validation = pca2.fit_transform(scale(validation_data))
print(pd.DataFrame(pca2.components_.T).head())

# -- Fit LR model: 6 PC's based on Elbow graph
lr_model_run_2.fit(data_reduced_validation[:,:6], validation_gspc_px)

# -- Run model:
predictions_3 = lr_model_run_2.predict(data_reduced_validation[:,:6])

# -- Find Metrics and Visualise:
mse_pred_3 = mean_squared_error(validation_gspc_px, predictions_3)
mae_pred_3 = mean_absolute_error(validation_gspc_px, predictions_3)
rmse_pred_3 = np.sqrt(mse_pred_3)
r2_3 = r2_score(validation_gspc_px, predictions_3)
median_ae_3 = median_absolute_error(validation_gspc_px, predictions_3)

print('Mean Squared Error:', mse_pred_3)
print('Mean Absolute Error:', mae_pred_3)
print('Root Mean Squared Error:', rmse_pred_3)
print('R-Squared:', r2_3)
print('Median Absolute Error:', median_ae_3)

# -- Compare results in table format:
df_compare_validation = pd.DataFrame({'ACTUAL_PRICE': validation_gspc_px, 'PREDICTED_PRICE': predictions_3.flatten()})
print(df_compare_validation.head(30))

plt.figure()
plot1, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['ACTUAL_PRICE'])
plot2, = plt.plot([i for i in range(0,len(df_compare_validation.index))], df_compare_validation['PREDICTED_PRICE'])
plt.xlabel('Prediction - Number of days')
plt.ylabel('Price of S&P500')
plt.title('Time Series - Compairson of Actual vs Predicted Prices')
plt.legend((plot1, plot2), ('S&P500 - Actual', 'S&P500 - Predicted'))

#########################################################
# Best subsets Regression:
#########################################################
# %matplotlib inline
import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time

# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.99):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.99):]

# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.99)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.99)]
data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data.reset_index(), non_validation_gspc.reset_index(), test_size=0.3, random_state=0)

del data_train['DATE']
del gspc_px_train['index']

def process_subset(feature_set_list):
    y = gspc_px_train
    X = data_train
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set_list)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set_list)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def best_model_per_predictor(k):
    X = data_train
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(process_subset(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc - tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

# Could take quite awhile to complete...
models_best = pd.DataFrame(columns=["RSS", "model"])

#x = best_model_per_predictor(2)

tic = time.time()
for i in range(1,11):
    #models_best.loc[i] = best_model_per_predictor(i)
    models_best['RSS'] = best_model_per_predictor(i)['RSS']
    models_best['model'] = best_model_per_predictor(i)['model']

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")


