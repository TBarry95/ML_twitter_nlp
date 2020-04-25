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
from sklearn.model_selection import LeaveOneOut
ion() # enables interactive mode

##########################################################################
#  EXTRACT: Read in cleaned dataset
##########################################################################
new_data_reduce3 = pd.read_csv(r".\quant_data_cleaned.csv")

###############################################
# Prepare data: Seperate dependent and independent variables
###############################################

gspc_px = new_data_reduce3['GSPC_CLOSE']
date = new_data_reduce3['DATE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['DATE']
del new_data_reduce3['GSPC_OPEN']
del new_data_reduce3['GSPC_LOW']
del new_data_reduce3['GSPC_ADJ_CLOSE']
del new_data_reduce3['GSPC_VOL']
del new_data_reduce3['GSPC_HIGH']

###############################################
# Cross validation K-Fold regression:
###############################################
# Necessary imports:
from sklearn.model_selection import cross_val_score, cross_val_predict

linear_mdl_cv = LinearRegression()
lr_kf_r2 = []
lr_kf_cvscore = []
for i in range(2,20):
    pred = cross_val_predict(linear_mdl_cv, new_data_reduce3, gspc_px, cv=i)
    cv_score = cross_val_score(linear_mdl_cv, new_data_reduce3, gspc_px, cv=i)
    lr_kf_r2.append(r2_score(gspc_px, pred))
    lr_kf_cvscore.append(cv_score)

plt.figure()
plt.plot([i for i in range(2,20)], lr_kf_r2)
plt.xlabel("Number of splits")
plt.ylabel("R-Squared")
plt.title("R-Squared per K number of splits")

pred_cv16 = cross_val_predict(linear_mdl_cv, new_data_reduce3, gspc_px, cv=16)
cross_val_score(linear_mdl_cv, new_data_reduce3, gspc_px, cv=8).mean()

from sklearn.model_selection import LeaveOneOut

X = new_data_reduce3
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))


pca_new = PCA()
reducer_all = pca_new.fit_transform(scale(new_data_reduce3))
lin_md = LinearRegression()

lr_kf_r2_pca = []
for i in range(2,20):
    pred = cross_val_predict(lin_md, reducer_all[:,20], gspc_px, cv=i)
    lr_kf_r2_pca.append(r2_score(gspc_px, pred))

from sklearn.linear_model import Ridge
data_train11, data_test11, gspc_px_train11, gspc_px_test11 = train_test_split(non_validation_data, non_validation_gspc, test_size=0.3, random_state=0, shuffle=True)
del data_train11['DATE']
del data_test11['DATE']

pc = PCA()
red = pc.fit_transform(data_train11)
redtest = pc.transform(data_test11)
redv = pc.transform(validation_data)
redtestv = pc.transform()