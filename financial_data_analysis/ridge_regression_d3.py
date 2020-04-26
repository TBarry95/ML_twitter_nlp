# Des: Using multiple quantitative and financial data points, this script conducts
#      regression analysis, PCA and and PCR.
# By: Tiernan Barry - x19141840 - NCI

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import *
from matplotlib.pyplot import ion
ion() # enables interactive mode
from sklearn.model_selection import GridSearchCV

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
#  Split datasets:
#########################################################

# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.99):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.99):]
# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.99)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.99)]

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0, shuffle=True)
val_date = validation_data['DATE']
del validation_data['DATE']
train_date = data_train['DATE']
test_date = data_test['DATE']
del data_train['DATE']
del data_test['DATE']

###############################################
#  Run Ridge regression using ALL predictors:
###############################################

print("# -- Ridge Cross Validation: -- #")
ridge_cv = Ridge()
params = {'alpha': [0.001, 0.01, 0.03, 0.05, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.4, 0.6, 0.8, 1, 5, 15, 20, 25, 30]}
ridge_reg = GridSearchCV(ridge_cv, params, scoring='neg_mean_squared_error', cv=20)
ridge_reg.fit(data_train, gspc_px_train)

print("Best alpha regularisation parameter: ", ridge_reg.best_params_)
print("Best MSE : ", ridge_reg.best_score_)

ridge_cv1 = Ridge(alpha=0.13)

ridge_cv1.fit(data_train, gspc_px_train)
ridge_cv1.predict(data_test)
ridge_cv1.score(data_test, gspc_px_test)
df_coef = pd.DataFrame()
df_coef['coef'] = [i for i in ridge_cv1.coef_]
df_coef['predictor'] = [i for i in data_train.columns]
print("Coefficients: ")
print(df_coef)

# -- Test regression using ALL predictors:
predictions_test = ridge_cv1.predict(data_test)

# -- Find Metrics and Visualise:
print("# -- Test Results - Ridge: All", len(data_train.columns), "Variables  -- #")
print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_test))
print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_test)))
print('R-Squared:', r2_score(gspc_px_test, predictions_test))
print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_test))
print("##########################################################")
print("##########################################################")

from yellowbrick.regressor import ResidualsPlot
plt.figure()
visualizer = ResidualsPlot(ridge_cv1)
visualizer.fit(data_train, gspc_px_train) # Fit the training data to the visualizer
visualizer.score(data_test, gspc_px_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure
print("Residuals distribution is notquite normal")

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_test.flatten()})
# print(df_compare.head(30))

###############################################
# 1. Validate Ridge regression using ALL predictors:
###############################################

print("# -- Validation Results - Ridge: All", len(data_train.columns), "Variables  -- #")

val_all_pred = ridge_cv1.predict(validation_data)

# -- Find Metrics and Visualise:
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

print("##########################################################")
print("##########################################################")


