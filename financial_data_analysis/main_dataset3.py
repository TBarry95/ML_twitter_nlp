# Des: Main executable file for DM and ML project deployment.
# By: Tiernan Barry - x19141840 - NCI

# 1. Run get dataset files:
print("1. EXPLORE DATA")
'''import financial_data_analysis.explore_and_clean_d3

import financial_data_analysis.correlation_mx_d3

import financial_data_analysis.linear_regression_d3

import financial_data_analysis.ridge_regression_d3

import financial_data_analysis.pca_linear_regression_d3

import financial_data_analysis.pca_ridge_regression_d3

#import financial_data_analysis.kfcv_linear'''
print("######################################################################################")
print("########## 1. EXPLORE DATA ########## ")
print("######################################################################################")
import explore_and_clean_d3

print("######################################################################################")
print("##########  2. CORRELATION MATRIX ########## ")
print("######################################################################################")
import correlation_mx_d3

print("######################################################################################")
print("##########  3. LINEAR REGRESSION ########## ")
print("######################################################################################")
import linear_regression_d3

print("######################################################################################")
print("##########  4. RIDGE REGRESSION ########## ")
print("######################################################################################")
import ridge_regression_d3

print("######################################################################################")
print("##########  5. PCA LINEAR REGRESSION ########## ")
print("######################################################################################")
import pca_linear_regression_d3

print("######################################################################################")
print("##########  6. PCA RIDGE REGRESSION ########## ")
print("######################################################################################")
import pca_ridge_regression_d3




