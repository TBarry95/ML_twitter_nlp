import quandl
import functions_nlp as fns
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import missingno as msno
from yahoo_finance import Share
import pandas as pd
#warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# # # # # # # # # # # # #
# Extract:
# # # # # # # # # # # # #

# 1. Get stock prices: SPX / GSPC
gspc_df = pd.read_csv(r"C:\Users\btier\Downloads\^GSPC.csv")
gspc_df.columns = ['DATE', 'GSPC_OPEN', 'GSPC_HIGH', 'GSPC_LOW', 'GSPC_CLOSE', 'GSPC_ADJ_CLOSE', 'GSPC_VOL']

# 2. Get predictors:
# -- GOLD:
gold_df = fns.get_quandl('LBMA/GOLD')
gold_df = gold_df.reset_index()
gold_df.columns = ['DATE', 'GOLD_USD_AM', 'GOLD_USD_PM', 'GOLD_GBP_AM', 'GOLD_GBP_PM', 'GOLD_EURO_AM', 'GOLD_EURO_PM']

# -- SILVER:
silver_df = fns.get_quandl('LBMA/SILVER')
silver_df = silver_df.reset_index()
silver_df.columns = ['DATE', 'SILVER_PRICE_USD', 'SILVER_PRICE_GBP', 'SILVER_PRICE_EUR']

# -- PLATINUM FUTURE:
platinum_future_df = fns.get_quandl('CHRIS/CME_PL1')
platinum_future_df = platinum_future_df.reset_index()
platinum_future_df.columns = ['DATE', 'PLAT_OPEN_USD', 'PLAT_HIGH_USD', 'PLAT_LOW_USD', 'PLAT_LAST_USD', 'PLAT_CHANGE', 'PLAT_SETTLE', 'PLAT_VOL', 'PLAT_PREV_DAY_OP_INT']

# -- PETROLEUM BASKET:
opec_basket = fns.get_quandl('OPEC/ORB') # OPEC Reference Basket
opec_basket = opec_basket.reset_index()
opec_basket.columns = ['DATE', 'OPEC_BSK_PRICE']

# -- NATURAL GAS:
natural_gas = fns.get_quandl('CHRIS/CME_NG1')
natural_gas = natural_gas.reset_index()
natural_gas.columns = ['DATE', 'GAS_OPEN_USD', 'GAS_HIGH_USD', 'GAS_LOW_USD', 'GAS_LAST_USD', 'GAS_CHANGE', 'GAS_SETTLE', 'GAS_VOL', 'GAS_PREV_DAY_OP_INT']

# -- EFFECTIVE FED FUND RATE DAILY :
interest_rates_daily = fns.get_quandl("FRED/DFF")
interest_rates_daily = interest_rates_daily.reset_index()
interest_rates_daily.columns = ['DATE', 'FED_FUND_RATE']

# -- 30-Year Treasury Constant Maturity Rate
treasury_30yr = fns.get_quandl("FRED/DGS30")
treasury_30yr = treasury_30yr.reset_index()
treasury_30yr.columns = ['DATE', '30YR_TRES_RATE']

# -- US GDP:
gdp_df = fns.get_quandl("FRED/GDP")
gdp_df = gdp_df.reset_index()
gdp_df.columns = ['DATE', 'GDP_USD_BILLION']

# -- USD/GBP:
usdgbp_df = fns.get_quandl("BOE/XUDLGBD")
usdgbp_df = usdgbp_df.reset_index()
usdgbp_df.columns = ['DATE', 'USD_GBP']

# -- Historical Housing Market Data - Real Building Cost Index
building_cost = fns.get_quandl("YALE/RBCI")
building_cost = building_cost.reset_index()
building_cost.columns = ['DATE', 'BUILD_COST_INX', 'US_POP_MILL', 'LONG_RATE']

# -- Stock Market Confidence Indices - United States Valuation Index Data - Institutional
confidence_inx_inst = fns.get_quandl("YALE/US_CONF_INDEX_VAL_INST")
confidence_inx_inst = confidence_inx_inst.reset_index()
confidence_inx_inst.columns = ['DATE', 'CONF_INX_INST', 'CONF_INX_ERROR']

# -- Stock Market Confidence Indices - United States Valuation Index Data - Institutional
confidence_inx_inst = fns.get_quandl("YALE/US_CONF_INDEX_VAL_INST")
confidence_inx_inst = confidence_inx_inst.reset_index()
confidence_inx_inst.columns = ['DATE', 'CONF_INX_INST', 'CONF_INX_ERROR']

# -- Historical Housing Market Data - Real Home Price Index
house_price = fns.get_quandl("YALE/RHPI")
house_price = house_price.reset_index()
house_price.columns = ['DATE', 'HOUSE_PX_INX_REAL']

# -- MERGE DATASETS:
all_data = pd.merge(gold_df, silver_df, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, platinum_future_df, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, opec_basket, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, natural_gas, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, interest_rates_daily, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, treasury_30yr, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, gdp_df, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, usdgbp_df, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, building_cost, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, confidence_inx_inst, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, house_price, how='left', left_on='DATE', right_on='DATE')
all_data['DATE'] = [str(i)[0:10] for i in all_data['DATE']]
all_data = pd.merge(all_data, gspc_df, how='left', left_on='DATE', right_on='DATE')

#all_data.to_csv(r"C:\Users\btier\Documents\economic_data.csv")

# # # # # # # # # # # # #
# Transform:
# # # # # # # # # # # # #

# 1. Check data: null values
msno.matrix(all_data, figsize= (50,30))

# Reduce - eg. 1 gold, 1 silver, etc..
new_data = pd.DataFrame({"DATE": all_data['DATE'],
                         "GOLD_USD_AM": all_data['GOLD_USD_AM'],
                         "SILVER_PRICE_USD": all_data['SILVER_PRICE_USD'],
                         "PLAT_OPEN_USD": all_data['PLAT_OPEN_USD'],
                         "FED_FUND_RATE": all_data['FED_FUND_RATE'],
                         "USD_GBP": all_data['USD_GBP'],
                         "BUILD_COST_INX": all_data['BUILD_COST_INX'],
                         "US_POP_MILL": all_data["US_POP_MILL"],
                         "CONF_INX_INST": all_data["CONF_INX_INST"],
                         "HOUSE_PX_INX_REAL": all_data["HOUSE_PX_INX_REAL"],
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
                         "HOUSE_PX_INX_REAL": new_data_reduce1["HOUSE_PX_INX_REAL"].fillna(method='ffill'),
                         "GSPC_CLOSE": new_data_reduce1['GSPC_CLOSE'].fillna(method='ffill')})

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
                         "HOUSE_PX_INX_REAL": new_data_reduce2["HOUSE_PX_INX_REAL"].fillna(method='bfill'),
                         "GSPC_CLOSE": new_data_reduce2['GSPC_CLOSE']})

# 4. Check data: clean dataset:
msno.matrix(new_data_reduce3)
sns.heatmap(new_data_reduce3.isnull(), cbar=False)

# Split data:
# -- Seperate dependent and ind variable
gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['DATE']

#  -- Validation: Keeping for last - never tested on
validation_data = new_data_reduce3[int(len(new_data_reduce3)*0.9):]
validation_gspc_px = gspc_px[int(len(gspc_px)*0.9):]

#  -- test / train on:
data_test_train = new_data_reduce3[:int(len(new_data_reduce3)*0.9)]
gspc_px_test_train = gspc_px[:int(len(gspc_px)*0.9)]

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(data_test_train, gspc_px_test_train, test_size=0.3, random_state=0)

data_train_fit = StandardScaler().fit_transform(data_train)
data_test_fit = StandardScaler().fit_transform(data_test)

pca = PCA()
data_train_pca = pca.fit_transform(data_train)
data_test_pca = pca.fit_transform(data_test)
exp_var = pca.explained_variance_ratio_

# test with 1 PC
pca = PCA(n_components=1)
data_train1 = pca.fit_transform(data_train)
data_test1 = pca.fit_transform(data_test)
exp_var1 = pca.explained_variance_ratio_

# test with 2 PC
pca = PCA(n_components=2)
data_train2 = pca.fit_transform(data_train)
data_test2 = pca.fit_transform(data_test)
exp_var2 = pca.explained_variance_ratio_

# test with 3 PC
pca = PCA(n_components=3)
data_train3 = pca.fit_transform(data_train)
data_test3 = pca.fit_transform(data_test)
exp_var3 = pca.explained_variance_ratio_

# test with 4 PC
pca = PCA(n_components=4)
data_train4 = pca.fit_transform(data_train)
data_test4 = pca.fit_transform(data_test)
exp_var4 = pca.explained_variance_ratio_

pc_df4 = pd.DataFrame(data_train4, columns = ['PC_1', 'PC_2', 'PC_3', 'PC_4'])
pc_df2 = pd.DataFrame(data_train2, columns = ['PC_1', 'PC_2'])

tst = pd.DataFrame({"PC1": pc_df2['PC_1'], "PC2": pc_df2['PC_2'], "PX": gspc_px_train})

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


