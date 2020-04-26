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

'''# -- Extract validation subset: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.85):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.85):]
# -- Test / Train split:
non_validation_data = new_data_reduce3[:int(len(new_data_reduce3)*0.85)]
non_validation_gspc = gspc_px[:int(len(gspc_px)*0.85)]

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0, shuffle=True)
val_date = validation_data['DATE']
del validation_data['DATE']'''

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.2, random_state=0, shuffle=True)

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

print("Optimal Number of Principal Components between 10-15 based on elbow graph")
print("##########################################################")
print("##########################################################")

#########################################################
# PCR: Principal Component Regression: Train and test (apply pca to both train and test)
#########################################################

for i in [1,2,3,4,5,6,7,8,10,12,14,16]:
    # -- Initialise PCA class
    pca3 = PCA()
    data_reduced_train = pca3.fit_transform(scale(data_train))
    data_reduced_test = pca3.fit_transform(scale(data_test))
    #v = pca3.fit_transform(scale(validation_data))
    # -- Initialise LR model
    lr_model_pca = LinearRegression()
    # -- Fit LR model: 6 PC's based on Elbow graph
    lr_model_pca.fit(data_reduced_train[:,:i], gspc_px_train)
    # -- Run model:
    predictions_2 = lr_model_pca.predict(data_reduced_test[:,:i])

    # -- Find Metrics and Visualise:
    print("# -- Test Results - PCR: ",i, " PCA Variables -- #")
    print('Mean Squared Error:', mean_squared_error(gspc_px_test, predictions_2))
    print('Mean Absolute Error:', mean_absolute_error(gspc_px_test, predictions_2))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(gspc_px_test, predictions_2)))
    print('R-Squared:', r2_score(gspc_px_test, predictions_2))
    print('Median Absolute Error:', median_absolute_error(gspc_px_test, predictions_2))
    print("##########################################################")
    print("##########################################################")

df = pd.DataFrame()
d = data_reduced_train[:,:15].reshape(15,8578)
df['pc1'] = d[0]
df['pc2'] = d[1]
df['pc3'] = d[2]
df['pc4'] = d[3]
df['pc5'] = d[4]
df['pc6'] = d[5]
df['pc7'] = d[6]
df['pc8'] = d[7]
df['pc9'] = d[8]
df['pc10'] = d[9]
df['pc11'] = d[10]
df['pc12'] = d[11]
df['pc13'] = d[12]
df['pc14'] = d[13]
corr_mx = df.corr()
mask_values = np.triu(np.ones_like(corr_mx, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
col_map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

###########################################
# residuals:
###########################################

from yellowbrick.regressor import ResidualsPlot
plt.figure()
visualizer = ResidualsPlot(lr_model_pca)
visualizer.fit(data_reduced_train[:,:16], gspc_px_train)  # Fit the training data to the visualizer
visualizer.score(data_reduced_test[:,:16], gspc_px_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))

'''
###########################################
# CV
###########################################

from sklearn.model_selection import cross_validate
scores_lst = []
for i in [2,3,4,5,6,7,8,10,12,14,16]:
    # -- Initialise PCA class
    pca3 = PCA()
    data_reduced_train = pca3.fit_transform(scale(data_train))
    data_reduced_test = pca3.fit_transform(scale(data_test))
    lrm = LinearRegression()
    scores = cross_validate(lrm, data_reduced_train[:,:], gspc_px_train,  cv=i, scoring=('r2'))
    scores_lst.append(scores)

'''



