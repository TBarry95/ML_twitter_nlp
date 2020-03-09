import quandl
import functions_nlp as fns
import numpy as np
from sklearn.model_selection import train_test_split


# # # # # # # # # # # # #
# Extract:
# # # # # # # # # # # # #

# 1. Get stock prices:
# -- SPX
spx_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("SPX"))
spx_target = spx_df['CLOSE_PRICE'].shift(-1) # get dependent variable (-1 = predict 1 day ahead, etc)
spx_validate = spx_df[0:int(len(spx_df)*0.10)]
spx_target_validate = spx_target[0:int(len(spx_df)*0.10)]
spx_df_test_train = spx_df[int(len(spx_df)*0.10):] # this to be split between test and train
spx_target_test_train = spx_target[int(len(spx_df)*0.10):]

# -- DJIA
djia_df = fns.alpha_v_to_df(fns.get_data_alpha_v2("DJIA"))
djia_target = djia_df['CLOSE_PRICE'].shift(-1)  # get dependent variable (-1 = predict 1 day ahead, etc)
djia_validate = djia_df[0:int(len(djia_df)*0.10)]
djia_target_validate = djia_target[0:int(len(djia_df)*0.10)]
djia_df_test_train = djia_df[int(len(djia_df)*0.10):] # this to be split between test and train
djia_target_test_train = djia_target[int(len(djia_target)*0.10):]

api_key = 'QK5pYuDbK7X6hZc9xj1x'
quandl.ApiConfig.api_key = api_key



data = quandl.get("LBMA/GOLD")