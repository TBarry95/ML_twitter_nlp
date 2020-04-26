# Des: Using multiple quantitative and financial data points, this script conducts
#      regression analysis, PCA and and PCR.
# By: Tiernan Barry - x19141840 - NCI

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import *
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.pyplot import ion
ion() # enables interactive mode

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

###############################################
# VIF: Variation Inflation Factor - Suggests which variables to keep.
###############################################

# -- temporarily remove date
date_tmp = new_data_reduce3['DATE']
del new_data_reduce3['DATE']

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

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.3, random_state=0, shuffle=True)

train_date = data_train['DATE']
test_date = data_test['DATE']
del data_train['DATE']
del data_test['DATE']

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
print("# -- Coefficients: -- #")

df = pd.DataFrame()
df['COEFS'] = [round(i,2) for i in lr_model_all_vars.coef_]
df['PREDICTOR'] = [i for i in data_train.columns]
print(df)

###############################################
# 2. Run OLS regression using VIF predictors:
###############################################
lr_model_vif_vars = LinearRegression()
lr_model_vif_vars.fit(data_train[[i for i in vif_factors]], gspc_px_train)
prediction_vif = lr_model_vif_vars.predict(data_test[[i for i in vif_factors]])
print(list(zip([round(i,4) for i in lr_model_vif_vars.coef_], data_train[[i for i in vif_factors]].columns)))

# -- Find Metrics and Visualise:
print("# -- Test Results - OLS: VIF Variables -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, prediction_vif))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, prediction_vif))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, prediction_vif)))
print('R-Squared:', r2_score(gspc_px_test, prediction_vif))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, prediction_vif))
print("VIF reduction does not show good potential")
print("##########################################################")
print("##########################################################")

###########################################
# residuals:
###########################################
from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(lr_model_all_vars)

visualizer.fit(data_train, gspc_px_train)  # Fit the training data to the visualizer
visualizer.score(data_test, gspc_px_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

print("Too many dimensions, need to reduce")
print("Coefficients are unstable - need to do PCA")
print("Residual distribution does not follow a proper normal distribution")