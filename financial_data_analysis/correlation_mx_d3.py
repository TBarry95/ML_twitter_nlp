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
sns.heatmap(corr_mx, mask=mask_values, cmap=col_map, center=0, annot=False,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print("Correlation Matrix reveals significant multicollinearity")
print("##########################################################")
print("##########################################################")

