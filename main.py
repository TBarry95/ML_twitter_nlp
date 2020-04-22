# Des: Main executable file for DM and ML project deployment.
# By: Tiernan Barry - x19141840 - NCI

# 1. Run get dataset files:

# 2. Clean datasets and print results to terminal:
print("###########################################################################################")
print("###########################################################################################")
print("# -- CLEAN RAW DATASETS -- #")
print("# -- Clean dataset 1: Media Twitter Data -- #")
import explore_dataset1_media

print("###########################################################################################")
print("# -- Clean dataset 2: Trump Twitter Data -- #")
import explore_dataset2_trump

print("###########################################################################################")
print("# -- Clean dataset 3: Financial/Economic Data -- #")
import explore_dataset3_quant

# 3. Get sentiments for dataset 1 and 2:
print("###########################################################################################")
print("###########################################################################################")
print("# -- SENTIMENTS ANALYSIS FOR DATASET 1 AND 2 -- #")
print("# -- Sentiment Analysis dataset 1: Media Twitter Data -- #")
import sentiment_dataset1_media

print("###########################################################################################")
print("# -- Sentiment Analysis dataset 2: Trump Twitter Data -- #")
import sentiment_dataset2_trump

print("###########################################################################################")
print("###########################################################################################")
print("# -- STOCK FORECASTING / PREDICTION -- #")

# 4. Stock Forecasting / Predictions:



