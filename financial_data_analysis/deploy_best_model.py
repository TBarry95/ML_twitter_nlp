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
from sklearn.linear_model import Ridge
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

gspc_px = new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_CLOSE']
del new_data_reduce3['GSPC_OPEN']
del new_data_reduce3['GSPC_LOW']
del new_data_reduce3['GSPC_ADJ_CLOSE']
del new_data_reduce3['GSPC_VOL']
del new_data_reduce3['GSPC_HIGH']
del new_data_reduce3['DATE']





pca3 = PCA()

data_train, data_test, gspc_px_train, gspc_px_test = train_test_split(new_data_reduce3, gspc_px, test_size=0.1, random_state=0, shuffle=False)

data_reduced_train = pca3.fit_transform(scale(data_train))
data_reduced_test = pca3.fit_transform(scale(data_test))
ridge_pcacv = Ridge(alpha=6, random_state=1)
ridge_pcacv.fit(data_reduced_train[:,:5], gspc_px_train)
ridge_pcacv.score(data_reduced_test[:,:5], gspc_px_test)

from yellowbrick.regressor import ResidualsPlot
plt.figure()
visualizer = ResidualsPlot(ridge_pcacv)
visualizer.fit(data_reduced_train[:,:5], gspc_px_train)  # Fit the training data to the visualizer
visualizer.score(data_reduced_test[:,:5], gspc_px_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

# -- Compare results in table format:
df_compare = pd.DataFrame({'ACTUAL_PRICE': gspc_px_test, 'PREDICTED_PRICE': predictions_2.flatten()})
# print(df_compare.head(30))