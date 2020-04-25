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
all_data = pd.read_csv(r".\economic_data.csv")

##########################################################################
# B. TRANSFORM: Clean data and prepare for analysis
##########################################################################
print("##########################################################")
print("# -- Data overview report: -- #")
print("Number of columns in raw dataset: ", len(all_data.columns))
print("Number of rows in raw dataset: ", len(all_data))
print("Date range of raw dataset: ", all_data['DATE'][-1:].values, "to ", all_data['DATE'][:1].values)
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
# -- get % of missing values for each column:
msno.matrix(all_data)
missing_val_summary = all_data.isna().mean()
percent_missing = (sum(missing_val_summary) / len(missing_val_summary) )*100
print("Missing values %: ", percent_missing)
missing_val_summary = pd.DataFrame(missing_val_summary)
missing_val_summary.plot(kind='bar', legend=False, title="Proportion of Missing Values: Economic/Financial Data")
missing_val_summary = missing_val_summary.reset_index()
missing_val_summary.columns = ['FIELD', 'MEAN']
missing_val_param = 0.2

new_data_cols = missing_val_summary['FIELD'][missing_val_summary['MEAN'] <= missing_val_param]
missing_data = missing_val_summary['FIELD'][missing_val_summary['MEAN'] > missing_val_param].values

# -- Reduce columns:
new_data = all_data[new_data_cols.values]
print("Dropped all columns which have more than", missing_val_param*100, "% missing values")
print("Columns dropped: ")
print(missing_data)
print("Reducing dataset from", len(all_data.columns), "columns to", len(new_data.columns))
missing_val_summary_1 = new_data.isna().mean()
percent_missing_1 = (sum(missing_val_summary_1) / len(missing_val_summary_1) )*100
print("Missing values %: ", percent_missing_1)
print("##########################################################")
print("##########################################################")

msno.matrix(new_data)
new_data_reduce1 = new_data[len(new_data['FNMA_CLOSE'])-len(new_data['FNMA_CLOSE'][new_data['FNMA_CLOSE'].notna()]):len(new_data['FNMA_CLOSE'])]
print("Cutting rows from ", len(new_data), " to ", len(new_data_reduce1), " rows")
print("##########################################################")
print("##########################################################")

missing_val_summary1 = new_data_reduce1.isna().mean()
percent_missing1 = (sum(missing_val_summary1) / len(missing_val_summary1) )*100
print("Missing values %: ", percent_missing1)

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

df_summary = new_data_reduce3.describe()

print("# -- Cleaned Data Set summary: Random Sample out of entire dataset (too big to print) -- #")
print(df_summary)
print("##########################################################")
print("##########################################################")

###############################################
# Export data
###############################################

new_data_reduce3.to_csv(r".\quant_data_cleaned.csv", index=False)

