# Des: Using multiple quantitative and financial data points, this script conducts
#      regression analysis, PCA and and PCR.
# By: Tiernan Barry - x19141840 - NCI

# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import missingno as msno
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import seaborn as sns
from matplotlib.pyplot import ion
ion() # enables interactive mode
from tabulate import tabulate

##########################################################################
# A. EXTRACT: Read in raw dataset as sourced from get_datasets.py
##########################################################################
all_data = pd.read_csv(r"C:\Users\btier\Documents\economic_data.csv")

##########################################################################
# B. TRANSFORM: Clean data and prepare for analysis
##########################################################################
print("##########################################################")
print("# -- Data overview report: -- #")
print("Number of columns in raw dataset: ", len(all_data.columns))
print("Number of rows in raw dataset: ", len(all_data))
print("##########################################################")
print("##########################################################")

print("# -- Column names in raw dataset: -- #")
print("Column names: ", all_data.columns)
print("##########################################################")
print("##########################################################")


###############################################
# 1. Check data: null values
###############################################
print("# -- Checking Null Values: -- #")
msno.matrix(all_data, figsize= (50,30))
# -- Reduce columns:
new_data = all_data.iloc[:, [0,1,2,3,4,7,8,10,11,12,13,15,16,17,27,28,30,40,42,43,44,45,46,47,48,49,50,51,54,55,56,57,
                             58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87]]
print("Reducing dataset from", len(all_data.columns), "columns to", len(new_data.columns), "due to lightly populated fields")
print("##########################################################")
print("##########################################################")

print("# -- Filtered column names: : -- #")
print("Column names: ", new_data.columns)
print("##########################################################")
print("##########################################################")

msno.matrix(new_data)
new_data_reduce1 = new_data[len(new_data['WMB_VOL'])-len(new_data['WMB_VOL'][new_data['WMB_VOL'].notna()]):len(new_data['WMB_VOL'])]

msno.matrix(new_data_reduce1)
new_data_reduce2 = new_data_reduce1.fillna(method='ffill')

msno.matrix(new_data_reduce2)

#####################################
# Impute values: Back fill: Appropriate for stock timeseries data
#####################################
new_data_reduce3 = new_data_reduce2
print("# -- Handling Null Values: -- #")
print("Where intermittent missing values exist, applying forward fill. ")
print("This is appropriate for stock prices due to weekend prices carrying forward, as well as other holidays ")
print("##########################################################")
print("##########################################################")

# -- Check data: clean dataset:
msno.matrix(new_data_reduce3)
sns.heatmap(new_data_reduce3.isnull(), cbar=False)

df_summary = new_data_reduce3[['GOLD_GBP_PM', 'SILVER_PRICE_USD',  'GSPC_CLOSE', 'LNC_CLOSE','JPM_CLOSE', 'JPM_VOL']].describe()

print("# -- Cleaned Data Set summary: Random Sample out of entire dataset (too big to print) -- #")
print(tabulate(df_summary, headers=df_summary.columns))
print("##########################################################")
print("##########################################################")

###############################################
# Export data
###############################################

new_data_reduce3.to_csv(r".\quant_data_cleaned.csv", index=False)

