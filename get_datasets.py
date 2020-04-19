# Des: Source script for extracting 2 out of the 3 required datasets for DM and ML project.
#      1. Twitter data from global media twitter pages.
#      2. Macroeconomic data from Quandl.
# By: Tiernan Barry - x19141840 - NCI.

# Libaries, imported files and installations (if required):
# pip install textblob
import functions_nlp as fns # Set of functions defined for this project
import warnings
import pandas as pd
import Twitter_API_Module as twt
warnings.simplefilter("ignore", DeprecationWarning)

######################################################################################
# Extract: 1. Twitter data from global media twitter pages.
######################################################################################

twitter_pgs = ["CNN", "BBCWorld", "BBCBreaking", "BBCNews", "ABC", "Independent",
               "RTENewsNow", "Independent_ie", "guardian", "guardiannews", "rtenews", "thejournal_ie",
               "wef", "IMFNews", "WHO", "euronews", "MailOnline", "TheSun", "Daily_Express", "DailyMirror",
               "standardnews", "LBC", "itvnews", "thetimes", "IrishTimes", "ANI", "XHNews", "TIME", "OANN",
               "BreitbartNews", "Channel4News", "BuzzFeedNews", "NewstalkFM", "NBCNewsBusiness", "CNBCnow",
               "markets", "YahooFinance", "MarketWatch", "Forbes", "businessinsider", "thehill", "CNNPolitics",
               "NPR", "AP", "USATODAY", "NYDailyNews", "nypost", "BBCLondonNews", "DailyMailUK",
               "CBSNews", "MSNBC", "nytimes", "FT", "business", "cnni", "RT_com", "AJEnglish", "CBS", "NewsHour",
               "BreakingNews", "cnnbrk", "WSJ", "Reuters", "SkyNews", "CBCAlerts"]

#tweets_list = fns.get_tweets_list(twitter_pgs, 120) #
tweets_list = fns.get_tweets_list(twitter_pgs, 1) # in reality, 120 pages per account are extracted. for demonstrating, takes 1 pages and ignores.
df_all_tweets = fns.tweets_to_df(tweets_list)
df_all_tweets = df_all_tweets.sort_values(by='DATE_TIME', ascending=0)
df_all_tweets = df_all_tweets.drop_duplicates()
# df_all_tweets.to_csv(r".\twitter_mass_media_data_2.csv", index= False)

######################################################################################
# Extract: 2. Macroeconomic data from Quandl.
######################################################################################

# 1. Get stock prices: SPX / GSPC
# -- manually downloaded from: https://finance.yahoo.com/quote/%5EGSPC/history?period1=-1325635200&period2=1586736000&interval=1d&filter=history&frequency=1d
gspc_df = pd.read_csv(r".\raw_datasets\^GSPC.csv")
gspc_df.columns = ['DATE', 'GSPC_OPEN', 'GSPC_HIGH', 'GSPC_LOW', 'GSPC_CLOSE', 'GSPC_ADJ_CLOSE', 'GSPC_VOL']

# 2. Get predictors:
PCAR = pd.read_csv(r".\raw_datasets\PCAR.csv")[['Date', 'Close', 'Volume']]
PCAR.columns = ['DATE', 'PCAR_CLOSE', 'PCAR_VOL']
WMT = pd.read_csv(r".\raw_datasets\WMT.csv")[['Date', 'Close', 'Volume']]
WMT.columns = ['DATE', 'WMT_CLOSE', 'WMT_VOL']
SWK = pd.read_csv(r".\raw_datasets\SWK.csv")[['Date', 'Close', 'Volume']]
SWK.columns = ['DATE', 'SWK_CLOSE', 'SWK_VOL']
BSX = pd.read_csv(r".\raw_datasets\BSX.csv")[['Date', 'Close', 'Volume']]
BSX.columns = ['DATE', 'BSX_CLOSE', 'BSX_VOL']
NI = pd.read_csv(r".\raw_datasets\NI.csv")[['Date', 'Close', 'Volume']]
NI.columns = ['DATE', 'NI_CLOSE', 'NI_VOL']
FNMA = pd.read_csv(r".\raw_datasets\FNMA.csv")[['Date', 'Close', 'Volume']]
FNMA.columns = ['DATE', 'FNMA_CLOSE', 'FNMA_VOL']
VFC = pd.read_csv(r".\raw_datasets\VFC.csv")[['Date', 'Close', 'Volume']]
VFC.columns = ['DATE', 'VFC_CLOSE', 'VFC_VOL']
TXT = pd.read_csv(r".\raw_datasets\TXT.csv")[['Date', 'Close', 'Volume']]
TXT.columns = ['DATE', 'TXT_CLOSE', 'TXT_VOL']
TGT = pd.read_csv(r".\raw_datasets\TGT.csv")[['Date', 'Close', 'Volume']]
TGT.columns = ['DATE', 'TGT_CLOSE', 'TGT_VOL']
INTC = pd.read_csv(r".\raw_datasets\INTC.csv")[['Date', 'Close', 'Volume']]
INTC.columns = ['DATE', 'INTC_CLOSE', 'INTC_VOL']
WFC = pd.read_csv(r".\raw_datasets\WFC.csv")[['Date', 'Close', 'Volume']]
WFC.columns = ['DATE', 'WFC_CLOSE', 'WFC_VOL']
DIS = pd.read_csv(r".\raw_datasets\DIS.csv")[['Date', 'Close', 'Volume']]
DIS.columns = ['DATE', 'DIS_CLOSE', 'DIS_VOL']
NEE = pd.read_csv(r".\raw_datasets\NEE.csv")[['Date', 'Close', 'Volume']]
NEE.columns = ['DATE', 'NEE_CLOSE', 'NEE_VOL']
TAP = pd.read_csv(r".\raw_datasets\TAP.csv")[['Date', 'Close', 'Volume']]
TAP.columns = ['DATE', 'TAP_CLOSE', 'TAP_VOL']
LNC = pd.read_csv(r".\raw_datasets\LNC.csv")[['Date', 'Close', 'Volume']]
LNC.columns = ['DATE', 'LNC_CLOSE', 'LNC_VOL']
DUK = pd.read_csv(r".\raw_datasets\DUK.csv")[['Date', 'Close', 'Volume']]
DUK.columns = ['DATE', 'DUK_CLOSE', 'DUK_VOL']
CI = pd.read_csv(r".\raw_datasets\CI.csv")[['Date', 'Close', 'Volume']]
CI.columns = ['DATE', 'CI_CLOSE', 'CI_VOL']
BAC = pd.read_csv(r".\raw_datasets\BAC.csv")[['Date', 'Close', 'Volume']]
BAC.columns = ['DATE', 'BAC_CLOSE', 'BAC_VOL']
IFF = pd.read_csv(r".\raw_datasets\IFF.csv")[['Date', 'Close', 'Volume']]
IFF.columns = ['DATE', 'IFF_CLOSE', 'IFF_VOL']
JPM = pd.read_csv(r".\raw_datasets\JPM.csv")[['Date', 'Close', 'Volume']]
JPM.columns = ['DATE', 'JPM_CLOSE', 'JPM_VOL']
WMB = pd.read_csv(r".\raw_datasets\WMB.csv")[['Date', 'Close', 'Volume']]
WMB.columns = ['DATE', 'WMB_CLOSE', 'WMB_VOL']

# -- BRENT: https://datahub.io/core/oil-prices#resource-brent-daily
brent_df = pd.read_csv(r".\raw_datasets\brent-daily_csv.csv")
brent_df.columns = ['DATE', 'BRENT_PRICE']

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

# -- Stock Market Confidence Indices - United States Valuation Index Data - Individual
confidence_inx_indv = fns.get_quandl("YALE/US_CONF_INDEX_VAL_INDIV")
confidence_inx_indv = confidence_inx_indv.reset_index()
confidence_inx_indv.columns = ['DATE', 'CONF_INX_INDV', 'CONF_INX_ERROR']

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
all_data = pd.merge(all_data, confidence_inx_indv, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, house_price, how='left', left_on='DATE', right_on='DATE')
all_data['DATE'] = [str(i)[0:10] for i in all_data['DATE']]
all_data = pd.merge(all_data, brent_df, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, gspc_df, how='left', left_on='DATE', right_on='DATE')

all_data = pd.merge(all_data, PCAR, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, WMT, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, SWK, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, BSX, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, NI, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, FNMA, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, VFC, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, TXT, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, INTC, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, WFC, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, DIS, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, NEE, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, TAP, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, LNC, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, DUK, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, TGT, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, CI, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, BAC, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, IFF, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, JPM, how='left', left_on='DATE', right_on='DATE')
all_data = pd.merge(all_data, WMB, how='left', left_on='DATE', right_on='DATE')

print("Raw dataset collected and exported to working directory")
all_data.to_csv(r".\economic_data.csv", index=False)

