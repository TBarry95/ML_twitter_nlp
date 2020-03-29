
# Libraries and source scripts:
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import missingno as msno
import pandas as pd
#warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import PCA
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import *

data = pd.read_csv(r"C:\Users\btier\Downloads\data_csv.csv")

sp_price = data['']

x_train, x_test, y_train, y_test = train_test_split(, , test_size=0.3, random_state=0)




