# House_Prices_Prediction
A complete work on Ames Housing dataset provided in the Kaggle competition - "House Prices - Advanced Regression Techniques".

https://iamshamikb.wordpress.com/2022/02/27/house-prices-prediction/

## Contents


```python
# Overview
# Imports
# Load train.csv
# Split the data into train, val, test
# Try a baseline RandomForest
# Reload original data
# Check for duplicates
# Check Multicolinearity
# Check which variables are really categorical
# Drop useless columns
# Fill missing values
# Check for outliers - not applied to final model
# Check skewed features - not applied to final model
# Check linear regression criterias
# Decide on columns for last time
# Try models
#     Linear regression
#         Vanialla
#         Elasticnet
#         Lasso
#         Ridge
#     Backward Feature Elimination
#         mlxtend
#         statsmodel
#     XGBRegressor
#         Try a baseline
#         Try Tuning
#     DTRegressor
#         Try a baseline
#         Try Tuning
#     Random Forest Regressor
#         Try a baseline
#         Try Tuning
```

## Overview

This notebook uses the Ames Housing dataset provided in the Kaggle competition - "House Prices - Advanced Regression Techniques".
<br> Link to competition page - 
<br> https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. 
<br> It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 
http://jse.amstat.org/v19n3/decock.pdf

Our objective is to predict the price of a house given the descriptive features of it.
<br> Target variable is - SalePrice and Metric is RMSE

The description given about metric on Kaggle page is this -

"Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)"

The submission format should be like this - 

Id,SalePrice
<br>1461,169000.1
<br>1462,187724.1233
<br>1463,175221
<br>etc.

The train and test data is given. As expected the test data does not have the saleprice column in it.
<br> Both Train and Test data has 80 feature columns, train has the target column too making it 81 for Train data.

Files given on Kaggle:
    
train.csv - the training set
<br>test.csv - the test set
<br>data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
<br>sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

Here's a brief version of what you'll find in the data description file.
<br>SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.<br>MSSubClass: The building class<br>MSZoning: The general zoning classification<br>LotFrontage: Linear feet of street connected to property<br>LotArea: Lot size in square feet<br>Street: Type of road access<br>Alley: Type of alley access<br>LotShape: General shape of property<br>LandContour: Flatness of the property<br>Utilities: Type of utilities available<br>LotConfig: Lot configuration<br>LandSlope: Slope of property<br>Neighborhood: Physical locations within Ames city limits<br>Condition1: Proximity to main road or railroad<br>Condition2: Proximity to main road or railroad (if a second is present)<br>BldgType: Type of dwelling<br>HouseStyle: Style of dwelling<br>OverallQual: Overall material and finish quality<br>OverallCond: Overall condition rating<br>YearBuilt: Original construction date<br>YearRemodAdd: Remodel date<br>RoofStyle: Type of roof<br>RoofMatl: Roof material<br>Exterior1st: Exterior covering on house<br>Exterior2nd: Exterior covering on house (if more than one material)<br>MasVnrType: Masonry veneer type<br>MasVnrArea: Masonry veneer area in square feet<br>ExterQual: Exterior material quality<br>ExterCond: Present condition of the material on the exterior<br>Foundation: Type of foundation<br>BsmtQual: Height of the basement<br>BsmtCond: General condition of the basement<br>BsmtExposure: Walkout or garden level basement walls<br>BsmtFinType1: Quality of basement finished area<br>BsmtFinSF1: Type 1 finished square feet<br>BsmtFinType2: Quality of second finished area (if present)<br>BsmtFinSF2: Type 2 finished square feet<br>BsmtUnfSF: Unfinished square feet of basement area<br>TotalBsmtSF: Total square feet of basement area<br>Heating: Type of heating<br>HeatingQC: Heating quality and condition<br>CentralAir: Central air conditioning<br>Electrical: Electrical system<br>1stFlrSF: First Floor square feet<br>2ndFlrSF: Second floor square feet<br>LowQualFinSF: Low quality finished square feet (all floors)<br>GrLivArea: Above grade (ground) living area square feet<br>BsmtFullBath: Basement full bathrooms<br>BsmtHalfBath: Basement half bathrooms<br>FullBath: Full bathrooms above grade<br>HalfBath: Half baths above grade<br>Bedroom: Number of bedrooms above basement level<br>Kitchen: Number of kitchens<br>KitchenQual: Kitchen quality<br>TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)<br>Functional: Home functionality rating<br>Fireplaces: Number of fireplaces<br>FireplaceQu: Fireplace quality<br>GarageType: Garage location<br>GarageYrBlt: Year garage was built<br>GarageFinish: Interior finish of the garage<br>GarageCars: Size of garage in car capacity<br>GarageArea: Size of garage in square feet<br>GarageQual: Garage quality<br>GarageCond: Garage condition<br>PavedDrive: Paved driveway<br>WoodDeckSF: Wood deck area in square feet<br>OpenPorchSF: Open porch area in square feet<br>EnclosedPorch: Enclosed porch area in square feet<br>3SsnPorch: Three season porch area in square feet<br>ScreenPorch: Screen porch area in square feet<br>PoolArea: Pool area in square feet<br>PoolQC: Pool quality<br>Fence: Fence quality<br>MiscFeature: Miscellaneous feature not covered in other categories<br>MiscVal: $Value of miscellaneous feature<br>MoSold: Month Sold<br>YrSold: Year Sold<br>SaleType: Type of sale<br>SaleCondition: Condition of sale

## Imports


```python
import warnings
warnings.filterwarnings("ignore")
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressorr̥
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
```

## Loading the data


```python
df = pd.read_csv('train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    

There are 1460 rows. Some of them have missing values.

## Try Baseline


```python
ndf = df.copy()

le = preprocessing.LabelEncoder()
ndf = ndf.apply(le.fit_transform)

features = ndf.drop(['SalePrice'], axis = 1)
target = ndf['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)

print('X_train: ', len(X_train), ' y_train: ', len(y_train))
print('X_val: ', len(X_val), ' y_val: ', len(y_val))
print('X_test: ', len(X_test), ' y_test: ', len(y_test))
```

    X_train:  1168  y_train:  1168
    X_val:  146  y_val:  146
    X_test:  146  y_test:  146
    


```python
ndf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype
    ---  ------         --------------  -----
     0   Id             1460 non-null   int64
     1   MSSubClass     1460 non-null   int64
     2   MSZoning       1460 non-null   int32
     3   LotFrontage    1460 non-null   int64
     4   LotArea        1460 non-null   int64
     5   Street         1460 non-null   int32
     6   Alley          1460 non-null   int32
     7   LotShape       1460 non-null   int32
     8   LandContour    1460 non-null   int32
     9   Utilities      1460 non-null   int32
     10  LotConfig      1460 non-null   int32
     11  LandSlope      1460 non-null   int32
     12  Neighborhood   1460 non-null   int32
     13  Condition1     1460 non-null   int32
     14  Condition2     1460 non-null   int32
     15  BldgType       1460 non-null   int32
     16  HouseStyle     1460 non-null   int32
     17  OverallQual    1460 non-null   int64
     18  OverallCond    1460 non-null   int64
     19  YearBuilt      1460 non-null   int64
     20  YearRemodAdd   1460 non-null   int64
     21  RoofStyle      1460 non-null   int32
     22  RoofMatl       1460 non-null   int32
     23  Exterior1st    1460 non-null   int32
     24  Exterior2nd    1460 non-null   int32
     25  MasVnrType     1460 non-null   int32
     26  MasVnrArea     1460 non-null   int64
     27  ExterQual      1460 non-null   int32
     28  ExterCond      1460 non-null   int32
     29  Foundation     1460 non-null   int32
     30  BsmtQual       1460 non-null   int32
     31  BsmtCond       1460 non-null   int32
     32  BsmtExposure   1460 non-null   int32
     33  BsmtFinType1   1460 non-null   int32
     34  BsmtFinSF1     1460 non-null   int64
     35  BsmtFinType2   1460 non-null   int32
     36  BsmtFinSF2     1460 non-null   int64
     37  BsmtUnfSF      1460 non-null   int64
     38  TotalBsmtSF    1460 non-null   int64
     39  Heating        1460 non-null   int32
     40  HeatingQC      1460 non-null   int32
     41  CentralAir     1460 non-null   int32
     42  Electrical     1460 non-null   int32
     43  1stFlrSF       1460 non-null   int64
     44  2ndFlrSF       1460 non-null   int64
     45  LowQualFinSF   1460 non-null   int64
     46  GrLivArea      1460 non-null   int64
     47  BsmtFullBath   1460 non-null   int64
     48  BsmtHalfBath   1460 non-null   int64
     49  FullBath       1460 non-null   int64
     50  HalfBath       1460 non-null   int64
     51  BedroomAbvGr   1460 non-null   int64
     52  KitchenAbvGr   1460 non-null   int64
     53  KitchenQual    1460 non-null   int32
     54  TotRmsAbvGrd   1460 non-null   int64
     55  Functional     1460 non-null   int32
     56  Fireplaces     1460 non-null   int64
     57  FireplaceQu    1460 non-null   int32
     58  GarageType     1460 non-null   int32
     59  GarageYrBlt    1460 non-null   int64
     60  GarageFinish   1460 non-null   int32
     61  GarageCars     1460 non-null   int64
     62  GarageArea     1460 non-null   int64
     63  GarageQual     1460 non-null   int32
     64  GarageCond     1460 non-null   int32
     65  PavedDrive     1460 non-null   int32
     66  WoodDeckSF     1460 non-null   int64
     67  OpenPorchSF    1460 non-null   int64
     68  EnclosedPorch  1460 non-null   int64
     69  3SsnPorch      1460 non-null   int64
     70  ScreenPorch    1460 non-null   int64
     71  PoolArea       1460 non-null   int64
     72  PoolQC         1460 non-null   int32
     73  Fence          1460 non-null   int32
     74  MiscFeature    1460 non-null   int32
     75  MiscVal        1460 non-null   int64
     76  MoSold         1460 non-null   int64
     77  YrSold         1460 non-null   int64
     78  SaleType       1460 non-null   int32
     79  SaleCondition  1460 non-null   int32
     80  SalePrice      1460 non-null   int64
    dtypes: int32(43), int64(38)
    memory usage: 678.8 KB
    


```python
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_train)

rms = sqrt(mean_squared_error(y_train, y_pred))
r2 = (r2_score(y_train, y_pred))
print('Train: ',rms, r2)

y_pred = regressor.predict(X_val)

rms = sqrt(mean_squared_error(y_val, y_pred))
r2 = (r2_score(y_val, y_pred))
print('Val: ',rms, r2)

# Train:  21.49352618964763 0.983618006494901
# Val:  49.14163049045355 0.9173903608878979
```

    Train:  21.49352618964763 0.983618006494901
    Val:  49.14163049045355 0.9173903608878979
    


```python
imp_df = pd.DataFrame({'features':X_train.columns, 'imp':regressor.feature_importances_}).sort_values(by=['imp'], ascending=False)
plt.figure(figsize=(10,20))
plt.xticks(rotation=45)
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.show()
```


    
![png](output_21_0.png)
    



```python
imp_df[:12]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>OverallQual</td>
      <td>0.586560</td>
    </tr>
    <tr>
      <th>46</th>
      <td>GrLivArea</td>
      <td>0.117834</td>
    </tr>
    <tr>
      <th>38</th>
      <td>TotalBsmtSF</td>
      <td>0.044983</td>
    </tr>
    <tr>
      <th>62</th>
      <td>GarageArea</td>
      <td>0.020556</td>
    </tr>
    <tr>
      <th>34</th>
      <td>BsmtFinSF1</td>
      <td>0.020272</td>
    </tr>
    <tr>
      <th>61</th>
      <td>GarageCars</td>
      <td>0.018131</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1stFlrSF</td>
      <td>0.017935</td>
    </tr>
    <tr>
      <th>58</th>
      <td>GarageType</td>
      <td>0.014508</td>
    </tr>
    <tr>
      <th>19</th>
      <td>YearBuilt</td>
      <td>0.013643</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LotArea</td>
      <td>0.012901</td>
    </tr>
    <tr>
      <th>60</th>
      <td>GarageFinish</td>
      <td>0.011543</td>
    </tr>
    <tr>
      <th>20</th>
      <td>YearRemodAdd</td>
      <td>0.009216</td>
    </tr>
  </tbody>
</table>
</div>



The model is suffering from overfitting and now we have some idea about which features might be important for us.
<br>We will clean the data and prepare for a Linear Regression model also we will check how features affect each other and the target variable more closely and in an interpretable manner.
<br>Also we will look into the missing values and do extensive EDA to learn more about the story behind the data.

## Reload the data


```python
df = pd.read_csv('train.csv')

features = df.drop(['SalePrice'], axis = 1)
target = df['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)

print('X_train: ', len(X_train), ' y_train: ', len(y_train))
print('X_val: ', len(X_val), ' y_val: ', len(y_val))
print('X_test: ', len(X_test), ' y_test: ', len(y_test))
```

    X_train:  1168  y_train:  1168
    X_val:  146  y_val:  146
    X_test:  146  y_test:  146
    


```python
X_train['SalePrice'] = y_train
df = X_train
```

## Duplicate observations


```python
len(df[df.duplicated()])
```




    0




```python
ndf = df.drop(['Id'], axis=1)
len(ndf[ndf.duplicated()])
```




    0



Confimed that we do not have any duplicate entries.

## Multicolinearity


```python
plt.figure(figsize=(15,10))
sns.heatmap(df.corr())
plt.show()
```


    
![png](output_32_0.png)
    



```python
for i in range(len(df.corr())):
    for j in range(len(df.corr())):
        if i<j:
            val = df.corr()[df.corr()r̥].iloc[i,j]
            if (val<1) and (val>0.7) :
                print(df.corr().columns[i],' ~ ',df.corr().columns[j])
```

    OverallQual  ~  SalePrice
    YearBuilt  ~  GarageYrBlt
    TotalBsmtSF  ~  1stFlrSF
    GrLivArea  ~  TotRmsAbvGrd
    GarageCars  ~  GarageArea
    

Most important features : OverallQual, GrLivArea  because they are correlated with target.

Features to choose from:
    
    YearBuilt  ~  GarageYrBlt
    TotalBsmtSF  ~  1stFlrSF
    GrLivArea  ~  TotRmsAbvGrd
    GarageCars  ~  GarageArea


```python
sns.pairplot(df[['YearBuilt', 'GarageYrBlt', 'TotalBsmtSF', 
                '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice']])
```




    <seaborn.axisgrid.PairGrid at 0x245a5fc2130>




    
![png](output_36_1.png)
    



```python
df[['YearBuilt', 'GarageYrBlt', 'TotalBsmtSF', 
    '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 8 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   YearBuilt     1168 non-null   int64  
     1   GarageYrBlt   1104 non-null   float64
     2   TotalBsmtSF   1168 non-null   int64  
     3   1stFlrSF      1168 non-null   int64  
     4   GrLivArea     1168 non-null   int64  
     5   TotRmsAbvGrd  1168 non-null   int64  
     6   GarageCars    1168 non-null   int64  
     7   SalePrice     1168 non-null   int64  
    dtypes: float64(1), int64(7)
    memory usage: 82.1 KB
    


```python
for i in ['YearBuilt', 'GarageYrBlt', 'TotalBsmtSF', 
    '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']:
    plt.title(i)
    df[i].hist()
    plt.show()
```


    
![png](output_38_0.png)
    



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    



    
![png](output_38_3.png)
    



    
![png](output_38_4.png)
    



    
![png](output_38_5.png)
    



    
![png](output_38_6.png)
    



    
![png](output_38_7.png)
    



    
![png](output_38_8.png)
    


Before choosing between two variables I want to check who has higher correlation with target and has better distribution and less no of nulls.


```python
high_corr_target = df.corr().SalePrice.sort_values()
for i in ['YearBuilt', 'GarageYrBlt', 'TotalBsmtSF', 
    '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']:
    print(i, round(high_corr_target[i], 2))
```

    YearBuilt 0.52
    GarageYrBlt 0.48
    TotalBsmtSF 0.6
    1stFlrSF 0.59
    GrLivArea 0.7
    TotRmsAbvGrd 0.52
    GarageCars 0.64
    GarageArea 0.62
    SalePrice 1.0
    

Deciding between variables which are highly correlated:
    
    YearBuilt  ~  GarageYrBlt -> I pick YearBuilt
    TotalBsmtSF  ~  1stFlrSF  -> I pick 1stFlrSF
    GrLivArea  ~  TotRmsAbvGrd-> I pick GrLivArea
    GarageCars  ~  GarageArea -> I pick GarageArea
    
Also we need to remember the variables with high correlation with the target:
    
    OverallQual, GrLivArea 

Let's see our choices:


```python
# 
for i in ['YearBuilt', '1stFlrSF', 'GrLivArea', 'GarageArea', 'OverallQual']:
    sns.scatterplot(data = df, x='SalePrice', y=i)
    plt.show()
```


    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    



    
![png](output_43_4.png)
    


If we look into OverallQual column, we will find that - houses with 8 and above rating starts at median price at 160000 around.
Houses with 9 rating starts at 200000 and with 10 rating at 300000, however the rating 10 has few outliers.


```python
df = df.drop(['GarageYrBlt', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GarageCars'], axis=1)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 77 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1168 non-null   int64  
     1   MSSubClass     1168 non-null   int64  
     2   MSZoning       1168 non-null   object 
     3   LotFrontage    951 non-null    float64
     4   LotArea        1168 non-null   int64  
     5   Street         1168 non-null   object 
     6   Alley          74 non-null     object 
     7   LotShape       1168 non-null   object 
     8   LandContour    1168 non-null   object 
     9   Utilities      1168 non-null   object 
     10  LotConfig      1168 non-null   object 
     11  LandSlope      1168 non-null   object 
     12  Neighborhood   1168 non-null   object 
     13  Condition1     1168 non-null   object 
     14  Condition2     1168 non-null   object 
     15  BldgType       1168 non-null   object 
     16  HouseStyle     1168 non-null   object 
     17  OverallQual    1168 non-null   int64  
     18  OverallCond    1168 non-null   int64  
     19  YearBuilt      1168 non-null   int64  
     20  YearRemodAdd   1168 non-null   int64  
     21  RoofStyle      1168 non-null   object 
     22  RoofMatl       1168 non-null   object 
     23  Exterior1st    1168 non-null   object 
     24  Exterior2nd    1168 non-null   object 
     25  MasVnrType     1162 non-null   object 
     26  MasVnrArea     1162 non-null   float64
     27  ExterQual      1168 non-null   object 
     28  ExterCond      1168 non-null   object 
     29  Foundation     1168 non-null   object 
     30  BsmtQual       1140 non-null   object 
     31  BsmtCond       1140 non-null   object 
     32  BsmtExposure   1140 non-null   object 
     33  BsmtFinType1   1140 non-null   object 
     34  BsmtFinSF1     1168 non-null   int64  
     35  BsmtFinType2   1140 non-null   object 
     36  BsmtFinSF2     1168 non-null   int64  
     37  BsmtUnfSF      1168 non-null   int64  
     38  Heating        1168 non-null   object 
     39  HeatingQC      1168 non-null   object 
     40  CentralAir     1168 non-null   object 
     41  Electrical     1167 non-null   object 
     42  1stFlrSF       1168 non-null   int64  
     43  2ndFlrSF       1168 non-null   int64  
     44  LowQualFinSF   1168 non-null   int64  
     45  GrLivArea      1168 non-null   int64  
     46  BsmtFullBath   1168 non-null   int64  
     47  BsmtHalfBath   1168 non-null   int64  
     48  FullBath       1168 non-null   int64  
     49  HalfBath       1168 non-null   int64  
     50  BedroomAbvGr   1168 non-null   int64  
     51  KitchenAbvGr   1168 non-null   int64  
     52  KitchenQual    1168 non-null   object 
     53  Functional     1168 non-null   object 
     54  Fireplaces     1168 non-null   int64  
     55  FireplaceQu    621 non-null    object 
     56  GarageType     1104 non-null   object 
     57  GarageFinish   1104 non-null   object 
     58  GarageArea     1168 non-null   int64  
     59  GarageQual     1104 non-null   object 
     60  GarageCond     1104 non-null   object 
     61  PavedDrive     1168 non-null   object 
     62  WoodDeckSF     1168 non-null   int64  
     63  OpenPorchSF    1168 non-null   int64  
     64  EnclosedPorch  1168 non-null   int64  
     65  3SsnPorch      1168 non-null   int64  
     66  ScreenPorch    1168 non-null   int64  
     67  PoolArea       1168 non-null   int64  
     68  PoolQC         6 non-null      object 
     69  Fence          233 non-null    object 
     70  MiscFeature    46 non-null     object 
     71  MiscVal        1168 non-null   int64  
     72  MoSold         1168 non-null   int64  
     73  YrSold         1168 non-null   int64  
     74  SaleType       1168 non-null   object 
     75  SaleCondition  1168 non-null   object 
     76  SalePrice      1168 non-null   int64  
    dtypes: float64(2), int64(32), object(43)
    memory usage: 711.8+ KB
    

So now we are left with 77 columns only.

We should drop the Id column rightaway as it doesn't help in anyway.


```python
df = df.drop(['Id'], axis=1)
```


```python
len(df.columns)
```




    76



## Check if potentially categorical


```python
pd.options.display.max_columns = 100
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>254</th>
      <td>20</td>
      <td>RL</td>
      <td>70.0</td>
      <td>8400</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1957</td>
      <td>1957</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>922</td>
      <td>Unf</td>
      <td>0</td>
      <td>392</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1314</td>
      <td>0</td>
      <td>0</td>
      <td>1314</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>294</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>145000</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>60</td>
      <td>RL</td>
      <td>59.0</td>
      <td>7837</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>7</td>
      <td>1993</td>
      <td>1994</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>799</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>799</td>
      <td>772</td>
      <td>0</td>
      <td>1571</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>380</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>178000</td>
    </tr>
    <tr>
      <th>638</th>
      <td>30</td>
      <td>RL</td>
      <td>67.0</td>
      <td>8777</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1910</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Fa</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>796</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>796</td>
      <td>0</td>
      <td>0</td>
      <td>796</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>P</td>
      <td>328</td>
      <td>0</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>799</th>
      <td>50</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>SWISU</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>7</td>
      <td>1937</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>252.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>569</td>
      <td>Unf</td>
      <td>0</td>
      <td>162</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>981</td>
      <td>787</td>
      <td>0</td>
      <td>1768</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>240</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>264</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>380</th>
      <td>50</td>
      <td>RL</td>
      <td>50.0</td>
      <td>5000</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>SWISU</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>6</td>
      <td>1924</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>LwQ</td>
      <td>218</td>
      <td>Unf</td>
      <td>0</td>
      <td>808</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1026</td>
      <td>665</td>
      <td>0</td>
      <td>1691</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>308</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>242</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>127000</td>
    </tr>
  </tbody>
</table>
</div>



From data dictionary I have decided to convert some columns to categorical.


```python
df[['OverallQual',   
    'OverallCond',  
    'YearBuilt',
    'YearRemodAdd',
    'Fireplaces',
    'MoSold',
    'YrSold']] = df[['OverallQual',   
                    'OverallCond',  
                    'YearBuilt',
                    'YearRemodAdd',
                    'Fireplaces',
                    'MoSold',
                    'YrSold']].astype(object)
# df.info()
```


```python
lens = []
for i in df.columns:
    lens.append(len(df[i].unique()))
plt.figure(figsize=(18,5))
plt.scatter(df.columns, lens)
plt.xticks(rotation=50)
plt.show()
```


    
![png](output_55_0.png)
    


As most columns have low no of distinct values, they are potentially categorical columns.


```python
for i in df.columns:
    print(i, len(df[i].unique()))
```

    MSSubClass 15
    MSZoning 5
    LotFrontage 108
    LotArea 890
    Street 2
    Alley 3
    LotShape 4
    LandContour 4
    Utilities 2
    LotConfig 5
    LandSlope 3
    Neighborhood 25
    Condition1 9
    Condition2 8
    BldgType 5
    HouseStyle 8
    OverallQual 10
    OverallCond 9
    YearBuilt 111
    YearRemodAdd 61
    RoofStyle 6
    RoofMatl 7
    Exterior1st 15
    Exterior2nd 16
    MasVnrType 5
    MasVnrArea 287
    ExterQual 4
    ExterCond 5
    Foundation 6
    BsmtQual 5
    BsmtCond 5
    BsmtExposure 5
    BsmtFinType1 7
    BsmtFinSF1 549
    BsmtFinType2 7
    BsmtFinSF2 118
    BsmtUnfSF 685
    Heating 6
    HeatingQC 5
    CentralAir 2
    Electrical 5
    1stFlrSF 657
    2ndFlrSF 361
    LowQualFinSF 20
    GrLivArea 734
    BsmtFullBath 4
    BsmtHalfBath 3
    FullBath 4
    HalfBath 3
    BedroomAbvGr 8
    KitchenAbvGr 4
    KitchenQual 4
    Functional 7
    Fireplaces 4
    FireplaceQu 6
    GarageType 7
    GarageFinish 4
    GarageArea 394
    GarageQual 6
    GarageCond 6
    PavedDrive 3
    WoodDeckSF 244
    OpenPorchSF 188
    EnclosedPorch 98
    3SsnPorch 17
    ScreenPorch 66
    PoolArea 7
    PoolQC 4
    Fence 5
    MiscFeature 5
    MiscVal 19
    MoSold 12
    YrSold 5
    SaleType 9
    SaleCondition 6
    SalePrice 571
    


```python
potentially_cat = []
for i in df.columns:
    if len(df[i].unique())<26:
        plt.title(i)
        df[i].hist()
        plt.show()
        potentially_cat.append(i)
```


    
![png](output_58_0.png)
    



    
![png](output_58_1.png)
    



    
![png](output_58_2.png)
    



    
![png](output_58_3.png)
    



    
![png](output_58_4.png)
    



    
![png](output_58_5.png)
    



    
![png](output_58_6.png)
    



    
![png](output_58_7.png)
    



    
![png](output_58_8.png)
    



    
![png](output_58_9.png)
    



    
![png](output_58_10.png)
    



    
![png](output_58_11.png)
    



    
![png](output_58_12.png)
    



    
![png](output_58_13.png)
    



    
![png](output_58_14.png)
    



    
![png](output_58_15.png)
    



    
![png](output_58_16.png)
    



    
![png](output_58_17.png)
    



    
![png](output_58_18.png)
    



    
![png](output_58_19.png)
    



    
![png](output_58_20.png)
    



    
![png](output_58_21.png)
    



    
![png](output_58_22.png)
    



    
![png](output_58_23.png)
    



    
![png](output_58_24.png)
    



    
![png](output_58_25.png)
    



    
![png](output_58_26.png)
    



    
![png](output_58_27.png)
    



    
![png](output_58_28.png)
    



    
![png](output_58_29.png)
    



    
![png](output_58_30.png)
    



    
![png](output_58_31.png)
    



    
![png](output_58_32.png)
    



    
![png](output_58_33.png)
    



    
![png](output_58_34.png)
    



    
![png](output_58_35.png)
    



    
![png](output_58_36.png)
    



    
![png](output_58_37.png)
    



    
![png](output_58_38.png)
    



    
![png](output_58_39.png)
    



    
![png](output_58_40.png)
    



    
![png](output_58_41.png)
    



    
![png](output_58_42.png)
    



    
![png](output_58_43.png)
    



    
![png](output_58_44.png)
    



    
![png](output_58_45.png)
    



    
![png](output_58_46.png)
    



    
![png](output_58_47.png)
    



    
![png](output_58_48.png)
    



    
![png](output_58_49.png)
    



    
![png](output_58_50.png)
    



    
![png](output_58_51.png)
    



    
![png](output_58_52.png)
    



    
![png](output_58_53.png)
    



    
![png](output_58_54.png)
    



    
![png](output_58_55.png)
    



    
![png](output_58_56.png)
    



    
![png](output_58_57.png)
    



    
![png](output_58_58.png)
    


Except MiscVal, everything is categorical. So we should convert them to object type.


```python
df[potentially_cat] = df[potentially_cat].astype(object)
```

Lets check how many cat and how many num columns are there at this point.


```python
print('Cat:', len(df.select_dtypes(include=[object]).columns))
print('Num:', len(df.select_dtypes(include=['int64', 'float64']).columns))
```

    Cat: 61
    Num: 15
    

## Drop useless categorical columns

Now lets check if any categorical column is useless. For example if we have a binary col with one column taking 90% of the values, then that column is useless, because there should be good split among categories.


```python
obj_type_cols = df.select_dtypes(include=[object]).columns
```


```python
critical_cols = []

for i in obj_type_cols:
    a = round(df[i].value_counts() / len(df) * 100)
    if any(i>80 for i in a):
        print(i)
        critical_cols.append(i)
```

    Street
    LandContour
    Utilities
    LandSlope
    Condition1
    Condition2
    BldgType
    RoofMatl
    ExterCond
    BsmtCond
    BsmtFinType2
    Heating
    CentralAir
    Electrical
    LowQualFinSF
    BsmtHalfBath
    KitchenAbvGr
    Functional
    GarageQual
    GarageCond
    PavedDrive
    3SsnPorch
    PoolArea
    MiscVal
    SaleType
    SaleCondition
    

So these are categorical columns with more than 80% values in one category.
<br>Let's check their distributions.


```python
for i in critical_cols:
    df[i].hist()
    plt.title(i)
    plt.show()
```


    
![png](output_68_0.png)
    



    
![png](output_68_1.png)
    



    
![png](output_68_2.png)
    



    
![png](output_68_3.png)
    



    
![png](output_68_4.png)
    



    
![png](output_68_5.png)
    



    
![png](output_68_6.png)
    



    
![png](output_68_7.png)
    



    
![png](output_68_8.png)
    



    
![png](output_68_9.png)
    



    
![png](output_68_10.png)
    



    
![png](output_68_11.png)
    



    
![png](output_68_12.png)
    



    
![png](output_68_13.png)
    



    
![png](output_68_14.png)
    



    
![png](output_68_15.png)
    



    
![png](output_68_16.png)
    



    
![png](output_68_17.png)
    



    
![png](output_68_18.png)
    



    
![png](output_68_19.png)
    



    
![png](output_68_20.png)
    



    
![png](output_68_21.png)
    



    
![png](output_68_22.png)
    



    
![png](output_68_23.png)
    



    
![png](output_68_24.png)
    



    
![png](output_68_25.png)
    


I will drop all these columns.


```python
df = df.drop(critical_cols, axis=1)
```


```python
print('Cat:', len(df.select_dtypes(include=[object]).columns))
print('Num:', len(df.select_dtypes(include=['int64', 'float64']).columns))
```

    Cat: 35
    Num: 15
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 50 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1168 non-null   object 
     1   MSZoning       1168 non-null   object 
     2   LotFrontage    951 non-null    float64
     3   LotArea        1168 non-null   int64  
     4   Alley          74 non-null     object 
     5   LotShape       1168 non-null   object 
     6   LotConfig      1168 non-null   object 
     7   Neighborhood   1168 non-null   object 
     8   HouseStyle     1168 non-null   object 
     9   OverallQual    1168 non-null   object 
     10  OverallCond    1168 non-null   object 
     11  YearBuilt      1168 non-null   object 
     12  YearRemodAdd   1168 non-null   object 
     13  RoofStyle      1168 non-null   object 
     14  Exterior1st    1168 non-null   object 
     15  Exterior2nd    1168 non-null   object 
     16  MasVnrType     1162 non-null   object 
     17  MasVnrArea     1162 non-null   float64
     18  ExterQual      1168 non-null   object 
     19  Foundation     1168 non-null   object 
     20  BsmtQual       1140 non-null   object 
     21  BsmtExposure   1140 non-null   object 
     22  BsmtFinType1   1140 non-null   object 
     23  BsmtFinSF1     1168 non-null   int64  
     24  BsmtFinSF2     1168 non-null   int64  
     25  BsmtUnfSF      1168 non-null   int64  
     26  HeatingQC      1168 non-null   object 
     27  1stFlrSF       1168 non-null   int64  
     28  2ndFlrSF       1168 non-null   int64  
     29  GrLivArea      1168 non-null   int64  
     30  BsmtFullBath   1168 non-null   object 
     31  FullBath       1168 non-null   object 
     32  HalfBath       1168 non-null   object 
     33  BedroomAbvGr   1168 non-null   object 
     34  KitchenQual    1168 non-null   object 
     35  Fireplaces     1168 non-null   object 
     36  FireplaceQu    621 non-null    object 
     37  GarageType     1104 non-null   object 
     38  GarageFinish   1104 non-null   object 
     39  GarageArea     1168 non-null   int64  
     40  WoodDeckSF     1168 non-null   int64  
     41  OpenPorchSF    1168 non-null   int64  
     42  EnclosedPorch  1168 non-null   int64  
     43  ScreenPorch    1168 non-null   int64  
     44  PoolQC         6 non-null      object 
     45  Fence          233 non-null    object 
     46  MiscFeature    46 non-null     object 
     47  MoSold         1168 non-null   object 
     48  YrSold         1168 non-null   object 
     49  SalePrice      1168 non-null   int64  
    dtypes: float64(2), int64(13), object(35)
    memory usage: 465.4+ KB
    

## Handling Missing Values


```python
missing_df = pd.DataFrame(df.isna().sum(), columns = ['No_of_missing'])
missing_df['feat_names'] = missing_df.index
missing_df['Percent_Missing'] = (missing_df['No_of_missing']/len(df))*100
missing_df = missing_df[missing_df['No_of_missing'] > 0]
missing_df = missing_df.sort_values(by = ['No_of_missing'])
missing_df['Non_null'] = len(df)-missing_df['No_of_missing']
missing_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No_of_missing</th>
      <th>feat_names</th>
      <th>Percent_Missing</th>
      <th>Non_null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MasVnrType</th>
      <td>6</td>
      <td>MasVnrType</td>
      <td>0.513699</td>
      <td>1162</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>6</td>
      <td>MasVnrArea</td>
      <td>0.513699</td>
      <td>1162</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>28</td>
      <td>BsmtQual</td>
      <td>2.397260</td>
      <td>1140</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>28</td>
      <td>BsmtExposure</td>
      <td>2.397260</td>
      <td>1140</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>28</td>
      <td>BsmtFinType1</td>
      <td>2.397260</td>
      <td>1140</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>64</td>
      <td>GarageType</td>
      <td>5.479452</td>
      <td>1104</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>64</td>
      <td>GarageFinish</td>
      <td>5.479452</td>
      <td>1104</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>217</td>
      <td>LotFrontage</td>
      <td>18.578767</td>
      <td>951</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>547</td>
      <td>FireplaceQu</td>
      <td>46.832192</td>
      <td>621</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>935</td>
      <td>Fence</td>
      <td>80.051370</td>
      <td>233</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1094</td>
      <td>Alley</td>
      <td>93.664384</td>
      <td>74</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1122</td>
      <td>MiscFeature</td>
      <td>96.061644</td>
      <td>46</td>
    </tr>
    <tr>
      <th>PoolQC</th>
      <td>1162</td>
      <td>PoolQC</td>
      <td>99.486301</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(missing_df)
```




    13




```python
plt.figure(figsize = (20,5))
sns.barplot(x = missing_df['feat_names'], y = missing_df['Percent_Missing'])
plt.xticks(rotation=25)
plt.title('Columns with missing values')
plt.xlabel('Col Names')
plt.ylabel('Percent data missing')
plt.show()
```


    
![png](output_76_0.png)
    



```python
plt.figure(figsize = (20,5))
sns.barplot(x = missing_df['feat_names'], y = missing_df['Non_null'])
plt.xticks(rotation=25)
plt.title('Columns with missing values')
plt.xlabel('Col Names')
plt.ylabel('Non Null Value Count')
plt.show()
```


    
![png](output_77_0.png)
    


Initially I think we should drop Fence, Alley, MiscFeature, PoolQC.
<br>But the data dictionary might have guidelines about the missing data.
<br>We will check what insights we can derive from them first.

Explanation of the columns which are having missing values:

MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
       
MasVnrArea: Masonry veneer area in square feet

BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
       
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
       
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
       
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage

GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage

LotFrontage: Linear feet of street connected to property

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
       
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
       
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
       
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
       
PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool

We  must tackle them one by one.

### MasVnrType

MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone


```python
df['MasVnrType'].hist()
df['MasVnrType'].value_counts(dropna=False)
```




    None       677
    BrkFace    366
    Stone      106
    BrkCmn      13
    NaN          6
    Name: MasVnrType, dtype: int64




    
![png](output_83_1.png)
    


I think the NaNs are nothing but Nones here, so will impute them with that.


```python
df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
```


```python
df['MasVnrType'].hist()
df['MasVnrType'].value_counts(dropna=False)
```




    None       683
    BrkFace    366
    Stone      106
    BrkCmn      13
    Name: MasVnrType, dtype: int64




    
![png](output_86_1.png)
    


### MasVnrArea

MasVnrArea: Masonry veneer area in square feet

This is a numerical column


```python
plt.figure(figsize=(20,5))
sns.histplot(data=df, x='MasVnrArea')
plt.show()
```


    
![png](output_89_0.png)
    



```python
df['MasVnrArea'].isnull().sum()
```




    6



We have 8 missing values here.


```python
vital_stat = df.describe().transpose().reset_index()
vital_stat[vital_stat['index'] == 'MasVnrArea']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>MasVnrArea</td>
      <td>1162.0</td>
      <td>103.771945</td>
      <td>173.032238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>166.0</td>
      <td>1378.0</td>
    </tr>
  </tbody>
</table>
</div>



From my understanding, masonry veneer means creating a fake brick wall on the actual concrete wall. 
<br>So there is a possiblity that most houses wont have it by default, because it is part of beautification.

Therefore it is totally ok to impute the missing values with 0.


```python
df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
```


```python
df['MasVnrArea'].isnull().sum()
```




    0



### BsmtQual

BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement


```python
df['BsmtQual'].hist()
df['BsmtQual'].value_counts(dropna=False)
```




    TA     521
    Gd     493
    Ex      97
    Fa      29
    NaN     28
    Name: BsmtQual, dtype: int64




    
![png](output_98_1.png)
    


The data dictionary says NA means no basement so we will impute these with a new category of No_Basement.


```python
df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'No_Basement'
```

### BsmtFinType1

BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement


```python
df['BsmtFinType1'].hist()
df['BsmtFinType1'].value_counts(dropna=False)
```




    Unf    345
    GLQ    328
    ALQ    178
    BLQ    123
    Rec    104
    LwQ     62
    NaN     28
    Name: BsmtFinType1, dtype: int64




    
![png](output_103_1.png)
    



```python
df.loc[df['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'No_Basement'
```

### BsmtExposure

BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement


```python
df['BsmtExposure'].hist()
df['BsmtExposure'].value_counts(dropna=False)
```




    No     769
    Av     175
    Gd     103
    Mn      93
    NaN     28
    Name: BsmtExposure, dtype: int64




    
![png](output_107_1.png)
    



```python
df.loc[df['BsmtExposure'].isnull(), 'BsmtExposure'] = 'No_Basement'
```

### GarageType

GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage


```python
df['GarageType'].hist()
df['GarageType'].value_counts(dropna=False)
```




    Attchd     693
    Detchd     308
    BuiltIn     74
    NaN         64
    Basment     16
    CarPort      7
    2Types       6
    Name: GarageType, dtype: int64




    
![png](output_111_1.png)
    


Just like basement, garage variables should be imputed with no garage.


```python
for i in ['GarageFinish',
         'GarageType']:
    df.loc[df[i].isnull(), i] = 'No_Garage'
```

### GarageFinis

GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage


```python
df['GarageFinish'].hist()
df['GarageFinish'].value_counts(dropna=False)
```




    Unf          480
    RFn          339
    Fin          285
    No_Garage     64
    Name: GarageFinish, dtype: int64




    
![png](output_116_1.png)
    


### LotFrontage

LotFrontage: Linear feet of street connected to property


```python
plt.figure(figsize=(20,5))
sns.histplot(data=df, x='LotFrontage')
plt.show()
```


    
![png](output_119_0.png)
    



```python
df['LotFrontage'].isnull().sum()
```




    217




```python
vital_stat[vital_stat['index'] == 'LotFrontage']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LotFrontage</td>
      <td>951.0</td>
      <td>70.343849</td>
      <td>24.897021</td>
      <td>21.0</td>
      <td>59.0</td>
      <td>70.0</td>
      <td>80.0</td>
      <td>313.0</td>
    </tr>
  </tbody>
</table>
</div>



Will impute with median.


```python
df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = df['LotFrontage'].median()
```

### FireplaceQu

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace


```python
df['FireplaceQu'].hist()
df['FireplaceQu'].value_counts(dropna=False)
```




    NaN    547
    Gd     305
    TA     252
    Fa      27
    Ex      21
    Po      16
    Name: FireplaceQu, dtype: int64




    
![png](output_126_1.png)
    


Will impute with No_Fireplace as given in data dictionary.


```python
df.loc[df['FireplaceQu'].isnull(), 'FireplaceQu'] = 'No Fireplace'
```

### Fence

Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence


```python
df['Fence'].hist()
df['Fence'].value_counts(dropna=False)
```




    NaN      935
    MnPrv    128
    GdPrv     50
    GdWo      46
    MnWw       9
    Name: Fence, dtype: int64




    
![png](output_131_1.png)
    


Impute with No_Fence


```python
df.loc[df['Fence'].isnull(), 'Fence'] = 'No_Fence'
```

### Alley

Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access


```python
df['Alley'].hist()
df['Alley'].value_counts(dropna=False)
```




    NaN     1094
    Grvl      44
    Pave      30
    Name: Alley, dtype: int64




    
![png](output_136_1.png)
    



```python
df.loc[df['Alley'].isnull(), 'Alley'] = 'No_Alley'
```

### MiscFeature

MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None


```python
df['MiscFeature'].hist()
df['MiscFeature'].value_counts(dropna=False)
```




    NaN     1122
    Shed      41
    Gar2       2
    Othr       2
    TenC       1
    Name: MiscFeature, dtype: int64




    
![png](output_140_1.png)
    


Impute with None


```python
df.loc[df['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
```

### PoolQC

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool


```python
df['PoolQC'].hist()
df['PoolQC'].value_counts(dropna=False)
```




    NaN    1162
    Ex        2
    Fa        2
    Gd        2
    Name: PoolQC, dtype: int64




    
![png](output_145_1.png)
    



```python
plt.figure(figsize=(15,5))
df['SalePrice'].hist(bins=50)
vital_stat[vital_stat['index'] == 'SalePrice']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>SalePrice</td>
      <td>1168.0</td>
      <td>181441.541952</td>
      <td>77263.583862</td>
      <td>34900.0</td>
      <td>130000.0</td>
      <td>165000.0</td>
      <td>214925.0</td>
      <td>745000.0</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_146_1.png)
    


So essentially the houses that got a pool has a price above median, mostly.


```python
df.loc[df['PoolQC'].isnull(), 'PoolQC'] = 'No_Pool'
```

### Check final results


```python
missing_df = pd.DataFrame(df.isna().sum(), columns = ['No_of_missing'])
missing_df['feat_names'] = missing_df.index
missing_df['Percent_Missing'] = (missing_df['No_of_missing']/len(df))*100
missing_df = missing_df[missing_df['No_of_missing'] > 0]
missing_df = missing_df.sort_values(by = ['No_of_missing'])
missing_df['Non_null'] = 1460-missing_df['No_of_missing']
missing_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No_of_missing</th>
      <th>feat_names</th>
      <th>Percent_Missing</th>
      <th>Non_null</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Dropping some more features

Alley, MiscFeature, PoolQC have too many values in NA category, they might not prove to be useful. So we should drop them.


```python
df = df.drop(['Alley', 'MiscFeature', 'PoolQC'], axis=1)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 47 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1168 non-null   object 
     1   MSZoning       1168 non-null   object 
     2   LotFrontage    1168 non-null   float64
     3   LotArea        1168 non-null   int64  
     4   LotShape       1168 non-null   object 
     5   LotConfig      1168 non-null   object 
     6   Neighborhood   1168 non-null   object 
     7   HouseStyle     1168 non-null   object 
     8   OverallQual    1168 non-null   object 
     9   OverallCond    1168 non-null   object 
     10  YearBuilt      1168 non-null   object 
     11  YearRemodAdd   1168 non-null   object 
     12  RoofStyle      1168 non-null   object 
     13  Exterior1st    1168 non-null   object 
     14  Exterior2nd    1168 non-null   object 
     15  MasVnrType     1168 non-null   object 
     16  MasVnrArea     1168 non-null   float64
     17  ExterQual      1168 non-null   object 
     18  Foundation     1168 non-null   object 
     19  BsmtQual       1168 non-null   object 
     20  BsmtExposure   1168 non-null   object 
     21  BsmtFinType1   1168 non-null   object 
     22  BsmtFinSF1     1168 non-null   int64  
     23  BsmtFinSF2     1168 non-null   int64  
     24  BsmtUnfSF      1168 non-null   int64  
     25  HeatingQC      1168 non-null   object 
     26  1stFlrSF       1168 non-null   int64  
     27  2ndFlrSF       1168 non-null   int64  
     28  GrLivArea      1168 non-null   int64  
     29  BsmtFullBath   1168 non-null   object 
     30  FullBath       1168 non-null   object 
     31  HalfBath       1168 non-null   object 
     32  BedroomAbvGr   1168 non-null   object 
     33  KitchenQual    1168 non-null   object 
     34  Fireplaces     1168 non-null   object 
     35  FireplaceQu    1168 non-null   object 
     36  GarageType     1168 non-null   object 
     37  GarageFinish   1168 non-null   object 
     38  GarageArea     1168 non-null   int64  
     39  WoodDeckSF     1168 non-null   int64  
     40  OpenPorchSF    1168 non-null   int64  
     41  EnclosedPorch  1168 non-null   int64  
     42  ScreenPorch    1168 non-null   int64  
     43  Fence          1168 non-null   object 
     44  MoSold         1168 non-null   object 
     45  YrSold         1168 non-null   object 
     46  SalePrice      1168 non-null   int64  
    dtypes: float64(2), int64(13), object(32)
    memory usage: 438.0+ KB
    

## Check outliers


```python
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
```


```python
# Reference: https://www.kaggle.com/nareshbhat/outlier-the-silent-killer
out=[]
def iqr_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df:
        if i > Upper_tail or i < Lower_tail:
            out.append(i)
    #     print("Outliers:",out)
    return out
cols_with_outliers = []
for i in num_cols:
    try:
        if len(iqr_outliers(df[i]))>0:
            cols_with_outliers.append(i)
    except:
        pass
cols_with_outliers
```




    ['LotFrontage',
     'LotArea',
     'MasVnrArea',
     'BsmtFinSF1',
     'BsmtFinSF2',
     'BsmtUnfSF',
     '1stFlrSF',
     '2ndFlrSF',
     'GrLivArea',
     'GarageArea',
     'WoodDeckSF',
     'OpenPorchSF',
     'EnclosedPorch',
     'ScreenPorch',
     'SalePrice']




```python
for i in cols_with_outliers:
    sns.boxplot(x=df[i])
    plt.show()
```


    
![png](output_158_0.png)
    



    
![png](output_158_1.png)
    



    
![png](output_158_2.png)
    



    
![png](output_158_3.png)
    



    
![png](output_158_4.png)
    



    
![png](output_158_5.png)
    



    
![png](output_158_6.png)
    



    
![png](output_158_7.png)
    



    
![png](output_158_8.png)
    



    
![png](output_158_9.png)
    



    
![png](output_158_10.png)
    



    
![png](output_158_11.png)
    



    
![png](output_158_12.png)
    



    
![png](output_158_13.png)
    



    
![png](output_158_14.png)
    


The below code is commented out because after removing the outliers with IQR and Median imputation, 
<br>models performed terribly. So decided to go without it as the variation in data will be lost if we do this.


```python
# # Reference: https://www.kaggle.com/nareshbhat/outlier-the-silent-killer
# def median_impute(col):
    
#     plt.figure(figsize=(15,5))
#     plt.subplot(1,2,1)
# #     sns.boxplot(df[col])
#     sns.histplot(df[col])
#     plt.title("Before")
# #     plt.show()

#     q1 = df[col].quantile(0.25)
#     q3 = df[col].quantile(0.75)
#     iqr = q3-q1
#     Lower_tail = q1 - 1.5 * iqr
#     Upper_tail = q3 + 1.5 * iqr

#     med = np.median(df[col])
#     percentile_90th = np.percentile(df[col], 90)
#     for i in df[col]:
#         if i > Upper_tail or i < Lower_tail:
# #                 df[col] = df[col].replace(i, med)
#                 df[col] = df[col].replace(i, percentile_90th)
    
#     plt.subplot(1,2,2)
# #     sns.boxplot(df[col])
#     sns.histplot(df[col])
#     plt.title("After")
#     plt.show()   
    
# for i in cols_with_outliers:
#     median_impute(i)
```


```python
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
```




    <AxesSubplot:>




    
![png](output_161_1.png)
    


As ScreenPorch has too many outliers, we will drop it.


```python
sns.histplot(df.ScreenPorch)
```




    <AxesSubplot:xlabel='ScreenPorch', ylabel='Count'>




    
![png](output_163_1.png)
    



```python
df = df.drop(['ScreenPorch'], axis=1)
```

## Transform Skewed Numerical Columns

Now I want to check the numerical columns.

We will transorm using lop1p if it is right skewed, otherwise with boxcox if the dist is in positive.


```python
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
```


```python
len(num_cols)
```




    14




```python
for i in num_cols:
    df[i].hist()
    plt.title(i)
    plt.show()
```


    
![png](output_170_0.png)
    



    
![png](output_170_1.png)
    



    
![png](output_170_2.png)
    



    
![png](output_170_3.png)
    



    
![png](output_170_4.png)
    



    
![png](output_170_5.png)
    



    
![png](output_170_6.png)
    



    
![png](output_170_7.png)
    



    
![png](output_170_8.png)
    



    
![png](output_170_9.png)
    



    
![png](output_170_10.png)
    



    
![png](output_170_11.png)
    



    
![png](output_170_12.png)
    



    
![png](output_170_13.png)
    


We have bimodal features and lots of skewed features. Only few are near normal.

Let's check the skewed features and transform them.


```python
skewed_cols = df[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
```


```python
skewed_cols
```




    LotArea          11.942726
    BsmtFinSF2        4.212476
    EnclosedPorch     3.159881
    LotFrontage       2.671465
    OpenPorchSF       2.328895
    MasVnrArea        2.291170
    BsmtFinSF1        1.859740
    SalePrice         1.740889
    WoodDeckSF        1.585291
    GrLivArea         1.423308
    1stFlrSF          1.420335
    BsmtUnfSF         0.909458
    2ndFlrSF          0.800180
    GarageArea        0.108970
    dtype: float64



We will transform features with skewness greater than 0.9.


```python
highly_skewed_cols = skewed_cols[abs(skewed_cols) > 0.9]
```


```python
highly_skewed_cols.index
```




    Index(['LotArea', 'BsmtFinSF2', 'EnclosedPorch', 'LotFrontage', 'OpenPorchSF',
           'MasVnrArea', 'BsmtFinSF1', 'SalePrice', 'WoodDeckSF', 'GrLivArea',
           '1stFlrSF', 'BsmtUnfSF'],
          dtype='object')




```python
for i in highly_skewed_cols.index:
    df[i].hist()
    plt.title(i)
    plt.show()
```


    
![png](output_178_0.png)
    



    
![png](output_178_1.png)
    



    
![png](output_178_2.png)
    



    
![png](output_178_3.png)
    



    
![png](output_178_4.png)
    



    
![png](output_178_5.png)
    



    
![png](output_178_6.png)
    



    
![png](output_178_7.png)
    



    
![png](output_178_8.png)
    



    
![png](output_178_9.png)
    



    
![png](output_178_10.png)
    



    
![png](output_178_11.png)
    


But let's check Target Variable separately.


```python
vital_stat = df.describe().transpose().reset_index()
vital_stat[vital_stat['index'] == 'SalePrice']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>SalePrice</td>
      <td>1168.0</td>
      <td>181441.541952</td>
      <td>77263.583862</td>
      <td>34900.0</td>
      <td>130000.0</td>
      <td>165000.0</td>
      <td>214925.0</td>
      <td>745000.0</td>
    </tr>
  </tbody>
</table>
</div>



Mean = 180921.19589
<br>Std Dev = 79442.502883


```python
sns.kdeplot(df.SalePrice)
plt.show()
```


    
![png](output_182_0.png)
    



```python
stats.probplot(df['SalePrice'], plot=plt)
plt.show()
```


    
![png](output_183_0.png)
    


We will not do this transformation so this code will remain commented.

A simple log(1+x) transformation should do the job.


```python
# for i in highly_skewed_cols.index:
#     print(i)
#     df[i] = np.log1p(df[i])
```

Lets see the transformed target variable.


```python
# sns.kdeplot(df.SalePrice)
# plt.show()
```


```python
# stats.probplot(df['SalePrice'], plot=plt)
# plt.show()
```

Lets see all the transformed variables.


```python
# for i in highly_skewed_cols.index:
#     df[i].hist()
#     plt.title(i)
#     plt.show()
```

## Checking criterias of Linear Regression

Linear Regression criterias:

    Independence of observations
    No or little Multicollinearity
    Relations between the independent and dependent variables must be linear
    
    Normality of the residuals
    Homoscedasticity
    No autocorrelation in the error term

### Check Independence

If feature definitions look like they can influence each other I will check their correlation and plots with each other to check for independence.

Check correlation:
    
    overallqual, overallcond
    yearbuild, yearremodadd
    Exterior1st, Exterior2nd    
    Foundation, Neighborhood
    BsmtFinSF1, BsmtFinSF2, BsmtUnfSF      
    1stFlrSF, 2ndFlrSF, GrLivArea , FullBath, HalfBath, BedroomAbvGr
    WoodDeckSF   
    OpenPorchSF, EnclosedPorch  
    YrSold, MoSold         


```python
cols_indep = ['OverallQual', 'OverallCond',
'YearBuilt', 'YearRemodAdd',
'Exterior1st', 'Exterior2nd' ,   
'Foundation', 'Neighborhood',
'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',      
'1stFlrSF', '2ndFlrSF', 'GrLivArea' , 'FullBath', 'HalfBath', 'BedroomAbvGr',
'WoodDeckSF',
'OpenPorchSF', 'EnclosedPorch',
'YrSold', 'MoSold']

sns.heatmap(df[cols_indep].corr())
```




    <AxesSubplot:>




    
![png](output_196_1.png)
    


I will look for correlation as low as 0.4 to get a broader view.


```python
for i in range(len(df[cols_indep].corr())):
    for j in range(len(df[cols_indep].corr())):
        if i<j:
            val = df[cols_indep].corr()[abs(df[cols_indep].corr())>0.4].iloc[i,j]
            if (val<1) and (abs(val)>0.4) :
                print(df[cols_indep].corr().columns[i],' ~ ',df[cols_indep].corr().columns[j])
```

    BsmtFinSF1  ~  BsmtUnfSF
    BsmtFinSF1  ~  1stFlrSF
    1stFlrSF  ~  GrLivArea
    2ndFlrSF  ~  GrLivArea
    

These relationships are concerning:

    YearBuilt  ~  YearRemodAdd
    BsmtFinSF1  ~  BsmtUnfSF
    BsmtFinSF1  ~  1stFlrSF
    1stFlrSF  ~  GrLivArea
    2ndFlrSF  ~  GrLivArea
    2ndFlrSF  ~  FullBath
    2ndFlrSF  ~  HalfBath
    2ndFlrSF  ~  BedroomAbvGr
    GrLivArea  ~  FullBath
    GrLivArea  ~  HalfBath
    GrLivArea  ~  BedroomAbvGr


```python
cols_indep = [
'YearBuilt', 'YearRemodAdd']

sns.pairplot(df[cols_indep])
```




    <seaborn.axisgrid.PairGrid at 0x245aafd96d0>




    
![png](output_200_1.png)
    



```python
cols_indep = [ 
'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']

sns.pairplot(df[cols_indep])
```




    <seaborn.axisgrid.PairGrid at 0x245ab0b3940>




    
![png](output_201_1.png)
    



```python
cols_indep = [    
'1stFlrSF', '2ndFlrSF', 'GrLivArea' , 'FullBath', 'HalfBath', 'BedroomAbvGr']

sns.pairplot(df[cols_indep])
```




    <seaborn.axisgrid.PairGrid at 0x245aaade7c0>




    
![png](output_202_1.png)
    



```python
cols_indep = [    
'OpenPorchSF', 'EnclosedPorch',
'YrSold', 'MoSold']

sns.pairplot(df[cols_indep])
```




    <seaborn.axisgrid.PairGrid at 0x245ad6d4e80>




    
![png](output_203_1.png)
    


Verdict

Keep:
    
    YearBuilt
    GrLivArea
    
Drop:
    
    1stfloor
    2ndfloor
    FullBath
    YearRemodAdd
    
Bad:
    
    YearBuilt-YearRemodAdd
    1stfloor - GrLivArea
    2ndfloor - GrLivArea
    2ndfloor- FullBath


```python
df = df.drop(['1stFlrSF', '2ndFlrSF', 'FullBath', 'YearRemodAdd'], axis=1)
```

### Check Linearity

We need to check if features are linear with target.


```python
for i in df.columns:
    sns.scatterplot(df[i], df['SalePrice'])
    plt.show()
```


    
![png](output_208_0.png)
    



    
![png](output_208_1.png)
    



    
![png](output_208_2.png)
    



    
![png](output_208_3.png)
    



    
![png](output_208_4.png)
    



    
![png](output_208_5.png)
    



    
![png](output_208_6.png)
    



    
![png](output_208_7.png)
    



    
![png](output_208_8.png)
    



    
![png](output_208_9.png)
    



    
![png](output_208_10.png)
    



    
![png](output_208_11.png)
    



    
![png](output_208_12.png)
    



    
![png](output_208_13.png)
    



    
![png](output_208_14.png)
    



    
![png](output_208_15.png)
    



    
![png](output_208_16.png)
    



    
![png](output_208_17.png)
    



    
![png](output_208_18.png)
    



    
![png](output_208_19.png)
    



    
![png](output_208_20.png)
    



    
![png](output_208_21.png)
    



    
![png](output_208_22.png)
    



    
![png](output_208_23.png)
    



    
![png](output_208_24.png)
    



    
![png](output_208_25.png)
    



    
![png](output_208_26.png)
    



    
![png](output_208_27.png)
    



    
![png](output_208_28.png)
    



    
![png](output_208_29.png)
    



    
![png](output_208_30.png)
    



    
![png](output_208_31.png)
    



    
![png](output_208_32.png)
    



    
![png](output_208_33.png)
    



    
![png](output_208_34.png)
    



    
![png](output_208_35.png)
    



    
![png](output_208_36.png)
    



    
![png](output_208_37.png)
    



    
![png](output_208_38.png)
    



    
![png](output_208_39.png)
    



    
![png](output_208_40.png)
    



    
![png](output_208_41.png)
    


The features dropped below are not linear with the target so it might help if we drop them.
<br> I wanted to drop features 'LotFrontage', 'BsmtFinSF1' too but baseline model found them important, so decided to keep them.


```python

# df = df.drop(['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF'], axis=1)
df = df.drop(['MasVnrArea', 'WoodDeckSF', 'OpenPorchSF'], axis=1)
```

## Final Columns Kept


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 39 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1168 non-null   object 
     1   MSZoning       1168 non-null   object 
     2   LotFrontage    1168 non-null   float64
     3   LotArea        1168 non-null   int64  
     4   LotShape       1168 non-null   object 
     5   LotConfig      1168 non-null   object 
     6   Neighborhood   1168 non-null   object 
     7   HouseStyle     1168 non-null   object 
     8   OverallQual    1168 non-null   object 
     9   OverallCond    1168 non-null   object 
     10  YearBuilt      1168 non-null   object 
     11  RoofStyle      1168 non-null   object 
     12  Exterior1st    1168 non-null   object 
     13  Exterior2nd    1168 non-null   object 
     14  MasVnrType     1168 non-null   object 
     15  ExterQual      1168 non-null   object 
     16  Foundation     1168 non-null   object 
     17  BsmtQual       1168 non-null   object 
     18  BsmtExposure   1168 non-null   object 
     19  BsmtFinType1   1168 non-null   object 
     20  BsmtFinSF1     1168 non-null   int64  
     21  BsmtFinSF2     1168 non-null   int64  
     22  BsmtUnfSF      1168 non-null   int64  
     23  HeatingQC      1168 non-null   object 
     24  GrLivArea      1168 non-null   int64  
     25  BsmtFullBath   1168 non-null   object 
     26  HalfBath       1168 non-null   object 
     27  BedroomAbvGr   1168 non-null   object 
     28  KitchenQual    1168 non-null   object 
     29  Fireplaces     1168 non-null   object 
     30  FireplaceQu    1168 non-null   object 
     31  GarageType     1168 non-null   object 
     32  GarageFinish   1168 non-null   object 
     33  GarageArea     1168 non-null   int64  
     34  EnclosedPorch  1168 non-null   int64  
     35  Fence          1168 non-null   object 
     36  MoSold         1168 non-null   object 
     37  YrSold         1168 non-null   object 
     38  SalePrice      1168 non-null   int64  
    dtypes: float64(1), int64(8), object(30)
    memory usage: 365.0+ KB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>LotShape</th>
      <th>LotConfig</th>
      <th>Neighborhood</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>RoofStyle</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>HeatingQC</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenQual</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>EnclosedPorch</th>
      <th>Fence</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>254</th>
      <td>20</td>
      <td>RL</td>
      <td>70.0</td>
      <td>8400</td>
      <td>Reg</td>
      <td>Inside</td>
      <td>NAmes</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1957</td>
      <td>Gable</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>922</td>
      <td>0</td>
      <td>392</td>
      <td>TA</td>
      <td>1314</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>TA</td>
      <td>0</td>
      <td>No Fireplace</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>294</td>
      <td>0</td>
      <td>No_Fence</td>
      <td>6</td>
      <td>2010</td>
      <td>145000</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>60</td>
      <td>RL</td>
      <td>59.0</td>
      <td>7837</td>
      <td>IR1</td>
      <td>Inside</td>
      <td>Gilbert</td>
      <td>2Story</td>
      <td>6</td>
      <td>7</td>
      <td>1993</td>
      <td>Gable</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>0</td>
      <td>799</td>
      <td>Gd</td>
      <td>1571</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>TA</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>380</td>
      <td>0</td>
      <td>No_Fence</td>
      <td>5</td>
      <td>2009</td>
      <td>178000</td>
    </tr>
    <tr>
      <th>638</th>
      <td>30</td>
      <td>RL</td>
      <td>67.0</td>
      <td>8777</td>
      <td>Reg</td>
      <td>Inside</td>
      <td>Edwards</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1910</td>
      <td>Gable</td>
      <td>MetalSd</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Fa</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>0</td>
      <td>796</td>
      <td>Gd</td>
      <td>796</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>TA</td>
      <td>0</td>
      <td>No Fireplace</td>
      <td>No_Garage</td>
      <td>No_Garage</td>
      <td>0</td>
      <td>164</td>
      <td>MnPrv</td>
      <td>5</td>
      <td>2008</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>799</th>
      <td>50</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Reg</td>
      <td>Corner</td>
      <td>SWISU</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>7</td>
      <td>1937</td>
      <td>Gable</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>569</td>
      <td>0</td>
      <td>162</td>
      <td>Ex</td>
      <td>1768</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>Gd</td>
      <td>2</td>
      <td>TA</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>240</td>
      <td>264</td>
      <td>MnPrv</td>
      <td>6</td>
      <td>2007</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>380</th>
      <td>50</td>
      <td>RL</td>
      <td>50.0</td>
      <td>5000</td>
      <td>Reg</td>
      <td>Inside</td>
      <td>SWISU</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>6</td>
      <td>1924</td>
      <td>Gable</td>
      <td>BrkFace</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>No</td>
      <td>LwQ</td>
      <td>218</td>
      <td>0</td>
      <td>808</td>
      <td>TA</td>
      <td>1691</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Gd</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>308</td>
      <td>242</td>
      <td>No_Fence</td>
      <td>5</td>
      <td>2010</td>
      <td>127000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols_kept = list(df.columns)
cols_kept
```




    ['MSSubClass',
     'MSZoning',
     'LotFrontage',
     'LotArea',
     'LotShape',
     'LotConfig',
     'Neighborhood',
     'HouseStyle',
     'OverallQual',
     'OverallCond',
     'YearBuilt',
     'RoofStyle',
     'Exterior1st',
     'Exterior2nd',
     'MasVnrType',
     'ExterQual',
     'Foundation',
     'BsmtQual',
     'BsmtExposure',
     'BsmtFinType1',
     'BsmtFinSF1',
     'BsmtFinSF2',
     'BsmtUnfSF',
     'HeatingQC',
     'GrLivArea',
     'BsmtFullBath',
     'HalfBath',
     'BedroomAbvGr',
     'KitchenQual',
     'Fireplaces',
     'FireplaceQu',
     'GarageType',
     'GarageFinish',
     'GarageArea',
     'EnclosedPorch',
     'Fence',
     'MoSold',
     'YrSold',
     'SalePrice']




```python
# cols_kept = ['MSSubClass',
#  'MSZoning',
#  'LotArea',
#  'LotShape',
#  'LotConfig',
#  'Neighborhood',
#  'HouseStyle',
#  'OverallQual',
#  'OverallCond',
#  'YearBuilt',
#  'RoofStyle',
#  'Exterior1st',
#  'Exterior2nd',
#  'MasVnrType',
#  'ExterQual',
#  'Foundation',
#  'BsmtQual',
#  'BsmtExposure',
#  'BsmtFinType1',
#  'BsmtFinSF2',
#  'BsmtUnfSF',
#  'HeatingQC',
#  'GrLivArea',
#  'BsmtFullBath',
#  'HalfBath',
#  'BedroomAbvGr',
#  'KitchenQual',
#  'Fireplaces',
#  'FireplaceQu',
#  'GarageType',
#  'GarageFinish',
#  'GarageArea',
#  'EnclosedPorch',
#  'Fence',
#  'MoSold',
#  'YrSold',
#  'SalePrice']
```

## A Dictionary to store model performances


```python
model_dict = {
    'vanilla_linear_regression':None,
    'ENreg':None,
    'lasso':None,
    'ridge':None,
    'lr_mlxtend':None,
    'regressor_OLS':None,
    'xgbr':None,
    'rfr':None
}
```

## Function for Scoring


```python
def print_score(model, train, val, y_train, y_val):
    
    y_pred = model.predict(train)
    rms1 = np.sqrt(mean_squared_error(y_train, y_pred))
    r2_1 = (r2_score(y_train, y_pred))
#     print(f'Train Errors: RMSE: {rms}, R Sq: {r2}')

    pred_on = val
    y_pred = model.predict(pred_on)
    rms2 = np.sqrt(mean_squared_error(y_val, y_pred))
    r2_2 = (r2_score(y_val, y_pred))
#     print(f'Val Errors: RMSE: {rms}, R Sq: {r2}')
    
    return rms1, rms2, r2_1, r2_2
```

## Linear Regression

As we have the test data available, we will merge all the data we have together and apply labelencoder on the categorical
<br>columns, then break the data into the previous train, val, test format.


```python
ndf = df.copy()

train = ndf.copy() #pd.read_csv('train.csv')[cols_kept]
test = pd.read_csv('test.csv')[cols_kept[:-1]]

df = pd.concat([train[cols_kept], X_val[cols_kept[:-1]], X_test[cols_kept[:-1]], test])

cat_cols = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood',
       'HouseStyle', 'YearBuilt',
       'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
       'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 
       'FireplaceQu', 'GarageType', 'GarageFinish', 'Fence', 
       'YrSold']

le = preprocessing.LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

print(len(df), len(X_val), len(X_test), len(train), len(test))

train = df.iloc[:len(train)]
X_val = df.iloc[len(train):len(train)+len(X_val)]
X_test = df.iloc[len(train)+len(X_val):len(train)+len(X_val)+len(X_test)]
test = df.iloc[len(train)+len(X_val)+len(X_test):]

# print(len(df), len(X_val), len(X_test), len(train), len(test))
# print(len(train.columns), len(X_val.columns), len(X_test.columns), len(test.columns))

for i in cols_kept[:-1]:
    X_val[i] = X_val[i].astype(dict(ndf.apply(lambda x: x.dtype))[i])
    X_test[i] = X_test[i].astype(dict(ndf.apply(lambda x: x.dtype))[i])
    
    
df = train.copy()

df.head()

X_train = df.drop(['SalePrice'], axis=1)
y_train = df['SalePrice']

# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# regressor.fit(X_train, y_train)

X_val = X_val.drop(['SalePrice'], axis=1)
X_test = X_test.drop(['SalePrice'], axis=1)

X_val.loc[X_val['LotFrontage'].isnull(), 'LotFrontage'] = X_train['LotFrontage'].median()
X_test.loc[X_test['LotFrontage'].isnull(), 'LotFrontage'] = X_train['LotFrontage'].median()
```

    2919 146 146 1168 1459
    

### Vanilla Linear Regression


```python
lr = LinearRegression(normalize=True)
# lr = LinearRegression(fit_intercept=False)

lr.fit(X_train, y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(lr, X_train, X_val[X_train.columns],  y_train, y_val)
tr_rmse, val_rmse, r2_1, r2_2 
```




    (32053.095397745885,
     25779.714268262076,
     0.8277488060019488,
     0.8865625924712075)




```python
vanilla_linear_regression = lr
model_dict['vanilla_linear_regression'] = [tr_rmse, val_rmse, r2_1, r2_2]
```


```python
y_train.mean(), y_train.median(), y_train.min()
```




    (181441.5419520548, 165000.0, 34900.0)




```python
lr.intercept_
```




    43472.82053329985




```python
imp_df = pd.DataFrame({'features':X_train.columns, 'imp':lr.coef_}).sort_values(by=['imp'], ascending=False)
# imp_df = imp_df[:12]
plt.figure(figsize=(5,25))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_228_0.png)
    


We need to check:
    
    Normality of the residuals
    Homoscedasticity
    No autocorrelation in the error term

Normality of the residuals - Pretty normal


```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_231_1.png)
    


Homoscedasticity - I see no cone shape, so no homoscedasticity.


```python
sns.scatterplot(y_pred, y_val - y_pred)
```




    <AxesSubplot:ylabel='SalePrice'>




    
![png](output_233_1.png)
    


Autocorrelation - No Autocorrelation


```python
# durbin_watson(y_val - y_pred)

# If this is within the range of 1.5 and 2.5, 
# we will consider there is no autocorrelation
```


```python
plot_acf(y_val - y_pred, alpha =0.05)
plt.show()
```


    
![png](output_236_0.png)
    



```python
plot_pacf(y_val - y_pred, alpha =0.05, lags=50)
plt.show()
```


    
![png](output_237_0.png)
    


### Elastic Net

Plot with alphas


```python
from sklearn.linear_model import ElasticNet


ENreg = ElasticNet(alpha=15, l1_ratio=0.5, normalize=False)

ENreg.fit(X_train, y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(ENreg, X_train, X_val[X_train.columns],  y_train, y_val)
model_dict['ENreg'] = [tr_rmse, val_rmse, r2_1, r2_2]
tr_rmse, val_rmse, r2_1, r2_2 
```




    (38316.81050445832, 32552.688825270074, 0.7538493476031198, 0.819126964499927)



### Lasso


```python
lassoReg = Lasso(normalize=True)

alphas = [0.001, 0.005, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1, 0.5, 0.9]
# lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03, 0.04, 0.05]}
lasso_params = {'alpha':alphas}

model = GridSearchCV(lassoReg, 
            param_grid=lasso_params).fit(X_train, y_train)

print(model.best_estimator_)

pred_on = X_train
y_pred = model.predict(pred_on)

r2 = (r2_score(y_train, y_pred))
print("Train Errors: ",rms, r2)

pred_on = X_val[X_train.columns]
y_pred = model.predict(pred_on)

rms = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = (r2_score(y_val, y_pred))
print("Val Errors: ",rms, r2)
```

    Lasso(alpha=0.9, normalize=True)
    Train Errors:  55.66391857228449 0.8277413610350386
    Val Errors:  25726.573834558396 0.8870297737606573
    


```python
val_rq_list = []
tr_rq_list = []

for i in alphas:
    
    model = Lasso(alpha = i, normalize=True)
    model.fit(X_train, y_train)

    pred_on = X_train
    y_pred = model.predict(pred_on)

    r2 = (r2_score(y_train, y_pred))
    # print("Val Errors: ",rms, r2)
    tr_rq_list.append(r2)
    
    pred_on = X_val[X_train.columns]
    y_pred = model.predict(pred_on)

    r2 = (r2_score(y_val, y_pred))
    # print("Val Errors: ",rms, r2)
    val_rq_list.append(r2)
    
plt.figure(figsize=(15,5))
sns.scatterplot(alphas, tr_rq_list, label ='Train R Sq')
sns.scatterplot(alphas, val_rq_list, label ='Val R Sq')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9e62b50d0>




    
![png](output_243_1.png)
    



```python
model = Lasso(alpha = 0.02, normalize=True)
model.fit(X_train, y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(model, X_train, X_val[X_train.columns], y_train, y_val)
model_dict['lasso'] = [tr_rmse, val_rmse, r2_1, r2_2]
print(tr_rmse, val_rmse, r2_1, r2_2 )

res_df = pd.DataFrame({'val':y_val, 'pred':y_pred})
sns.scatterplot(res_df.val, res_df.pred)
```

    32053.095747705964 25778.509259490234 0.8277488022406245 0.8865731969236088
    




    <AxesSubplot:xlabel='val', ylabel='pred'>




    
![png](output_244_2.png)
    



```python
lasso = model
```


```python
model.intercept_
```




    43460.2874056717




```python
imp_df = pd.DataFrame({'features':X_train.columns, 'imp':model.coef_}).sort_values(by=['imp'], ascending=False)
# imp_df = imp_df[:12]
plt.figure(figsize=(5,25))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_247_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_248_1.png)
    



```python
sns.scatterplot(y_pred, y_val - y_pred)
```




    <AxesSubplot:ylabel='SalePrice'>




    
![png](output_249_1.png)
    



```python
durbin_watson(y_val - y_pred)

# If this is within the range of 1.5 and 2.5, 
# we will consider there is no autocorrelation
```




    2.049293302718209




```python
plot_acf(y_val - y_pred, alpha =0.05)
plt.show()
```


    
![png](output_251_0.png)
    



```python
plot_pacf(y_val - y_pred, alpha =0.05, lags=50)
plt.show()
```


    
![png](output_252_0.png)
    


### Ridge


```python
ridgeReg = Ridge(normalize=True)

ridge_params = {'alpha':[0.05, 0.08, 0.1, 0.5, 1, 5, 10, 200, 230, 250,265, 270, 275, 290, 300, 500, 1000]}

model = GridSearchCV(ridgeReg, 
            param_grid=ridge_params).fit(X_train, y_train)

print(model.best_estimator_)

pred_on = X_train
y_pred = model.predict(pred_on)

r2 = (r2_score(y_train, y_pred))
print("Train Errors: ",rms, r2)

pred_on = X_val[X_train.columns]
y_pred = model.predict(pred_on)

rms = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = (r2_score(y_val, y_pred))
print("Val Errors: ",rms, r2)
```

    Ridge(alpha=0.1, normalize=True)
    Train Errors:  25726.573834558396 0.8251230437768077
    Val Errors:  25665.740342350215 0.8875634047180159
    


```python
val_rq_list = []
tr_rq_list = []
alphas = [0.025, 0.05, 0.08, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 50]
for i in alphas:
    
    model = Ridge(alpha = i, normalize=True)
    model.fit(X_train, y_train)

    pred_on = X_train
    y_pred = model.predict(pred_on)

    r2 = (r2_score(y_train, y_pred))
    # print("Val Errors: ",rms, r2)
    tr_rq_list.append(r2)
    
    pred_on = X_val[X_train.columns]
    y_pred = model.predict(pred_on)

    r2 = (r2_score(y_val, y_pred))
    # print("Val Errors: ",rms, r2)
    val_rq_list.append(r2)
    
plt.figure(figsize=(15,5))
sns.scatterplot(alphas, tr_rq_list, label ='Train R Sq')
sns.scatterplot(alphas, val_rq_list, label ='Val R Sq')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9e7728190>




    
![png](output_255_1.png)
    


So I would say that alpha 0.1 is a good parameter for Ridge Regression and the model did not overfit with it.


```python
model = Ridge(alpha = 0.1, normalize=True)
model.fit(X_train, y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(model, X_train, X_val[X_train.columns], y_train, y_val)
model_dict['ridge'] = [tr_rmse, val_rmse, r2_1, r2_2]
print(tr_rmse, val_rmse, r2_1, r2_2 )
```

    32296.476830362637 25665.740342350215 0.8251230437768077 0.8875634047180159
    


```python
ridge = model
```


```python
model.intercept_
```




    53411.1232459592




```python
imp_df = pd.DataFrame({'features':X_train.columns, 'imp':model.coef_}).sort_values(by=['imp'], ascending=False)
# imp_df = imp_df[:12]
plt.figure(figsize=(5,25))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_260_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_261_1.png)
    



```python
sns.scatterplot(y_pred, y_val - y_pred)
```




    <AxesSubplot:ylabel='SalePrice'>




    
![png](output_262_1.png)
    



```python
durbin_watson(y_val - y_pred)

# If this is within the range of 1.5 and 2.5, 
# we will consider there is no autocorrelation
```




    1.9141660141495336




```python
plot_acf(y_val - y_pred, alpha =0.05)
plt.show()
```


    
![png](output_264_0.png)
    



```python
plot_pacf(y_val - y_pred, alpha =0.05, lags=50)
plt.show()
```


    
![png](output_265_0.png)
    


### Linear Regression - Backward Feature Elimination

#### Using mlextend

I wanted to check 20 most important features, we can increase this no in an extension to this project.


```python
# !pip install mlxtend
```


```python
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

lreg = LinearRegression()

sfs1 = sfs(lreg, k_features=20, forward=False, verbose=0, scoring='neg_mean_squared_error')

sfs1 = sfs1.fit(X_train, y_train)

feat_names = list(sfs1.k_feature_names_)
```


```python
print(feat_names)
```

    ['MSSubClass', 'LotArea', 'LotShape', 'Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'GrLivArea', 'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageFinish', 'GarageArea']
    


```python
new_data = X_train[feat_names]
new_data['SalePrice'] = y_train

new_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotArea</th>
      <th>LotShape</th>
      <th>Neighborhood</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>RoofStyle</th>
      <th>ExterQual</th>
      <th>BsmtQual</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>KitchenQual</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>254</th>
      <td>20</td>
      <td>8400</td>
      <td>3</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>64</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>1314</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>294.0</td>
      <td>145000.0</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>60</td>
      <td>7837</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>100</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1571</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>380.0</td>
      <td>178000.0</td>
    </tr>
    <tr>
      <th>638</th>
      <td>30</td>
      <td>8777</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>796</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>85000.0</td>
    </tr>
    <tr>
      <th>799</th>
      <td>50</td>
      <td>7200</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>46</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1768</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>240.0</td>
      <td>175000.0</td>
    </tr>
    <tr>
      <th>380</th>
      <td>50</td>
      <td>5000</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>34</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>1691</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>308.0</td>
      <td>127000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = LinearRegression(normalize=True)
model.fit(X_train[feat_names], y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(model, X_train[feat_names], X_val[feat_names], y_train, y_val)
model_dict['lr_mlxtend'] = [tr_rmse, val_rmse, r2_1, r2_2]
print(tr_rmse, val_rmse, r2_1, r2_2 )

res_df = pd.DataFrame({'val':y_val, 'pred':y_pred})
sns.scatterplot(res_df.val, res_df.pred)
```

    32407.801010332543 25693.641780058737 0.8239153834818104 0.8873188103424505
    




    <AxesSubplot:xlabel='val', ylabel='pred'>




    
![png](output_273_2.png)
    



```python
lr_mlxtend = model
```


```python
model.intercept_
```




    19616.89168432253




```python
imp_df = pd.DataFrame({'features':feat_names, 'imp':model.coef_}).sort_values(by=['imp'], ascending=False)
# imp_df = imp_df[:12]
plt.figure(figsize=(5,5))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_276_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_277_1.png)
    



```python
sns.scatterplot(y_pred, y_val - y_pred)
```




    <AxesSubplot:ylabel='SalePrice'>




    
![png](output_278_1.png)
    



```python
durbin_watson(y_val - y_pred)

# If this is within the range of 1.5 and 2.5, 
# we will consider there is no autocorrelation
```




    1.9141660141495336




```python
plot_acf(y_val - y_pred, alpha =0.05)
plt.show()
```


    
![png](output_280_0.png)
    



```python
plot_pacf(y_val - y_pred, alpha =0.05, lags=50)
plt.show()
```


    
![png](output_281_0.png)
    


#### Using statsmodels


```python
X_train_copy = X_train.copy()

condition = True
# condition = False
while(condition):
    X_train_opt = np.append(arr = np.ones((len(X_train),1)).astype(float), values = X_train_copy, axis = 1) 

    regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt.astype(float)).fit()

    pvals = regressor_OLS.pvalues.iloc[1:]
    if input()=='n':
        condition = False
    else:
        if max(pvals)>0.05:
            indx_to_remove = pvals.argmax()
            X_train_copy = X_train_copy.drop(X_train_copy.columns[indx_to_remove], axis=1)
        else:
            print('Done.')
            condition = False
    print(pd.read_html(regressor_OLS.summary().tables[0].as_html(),header=0,index_col=0)[0])
    
regressor_OLS.summary()
```

    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.822
    Method:               Least Squares         F-statistic:    142.800
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:34      Log-Likelihood: -13775.000
    No. Observations:              1168                 AIC:  27630.000
    Df Residuals:                  1129                 BIC:  27830.000
    Df Model:                        38                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.822
    Method:               Least Squares         F-statistic:    146.800
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:34      Log-Likelihood: -13775.000
    No. Observations:              1168                 AIC:  27630.000
    Df Residuals:                  1130                 BIC:  27820.000
    Df Model:                        37                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.822
    Method:               Least Squares         F-statistic:    151.000
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:35      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27630.000
    Df Residuals:                  1131                 BIC:  27810.000
    Df Model:                        36                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.822
    Method:               Least Squares         F-statistic:    155.400
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:35      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27620.000
    Df Residuals:                  1132                 BIC:  27810.000
    Df Model:                        35                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    160.100
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:36      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27620.000
    Df Residuals:                  1133                 BIC:  27800.000
    Df Model:                        34                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    165.100
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:36      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27620.000
    Df Residuals:                  1134                 BIC:  27790.000
    Df Model:                        33                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    170.400
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:37      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27620.000
    Df Residuals:                  1135                 BIC:  27780.000
    Df Model:                        32                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    176.000
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:37      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27620.000
    Df Residuals:                  1136                 BIC:  27780.000
    Df Model:                        31                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    182.000
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:37      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1137                 BIC:  27770.000
    Df Model:                        30                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.828
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    188.300
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:37      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1138                 BIC:  27760.000
    Df Model:                        29                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    195.100
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:37      Log-Likelihood: -13776.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1139                 BIC:  27760.000
    Df Model:                        28                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    202.400
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:38      Log-Likelihood: -13777.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1140                 BIC:  27750.000
    Df Model:                        27                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    210.300
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:38      Log-Likelihood: -13777.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1141                 BIC:  27740.000
    Df Model:                        26                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.823
    Method:               Least Squares         F-statistic:    218.700
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:38      Log-Likelihood: -13777.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1142                 BIC:  27740.000
    Df Model:                        25                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.824
    Method:               Least Squares         F-statistic:    227.900
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:39      Log-Likelihood: -13778.000
    No. Observations:              1168                 AIC:  27610.000
    Df Residuals:                  1143                 BIC:  27730.000
    Df Model:                        24                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.824
    Method:               Least Squares         F-statistic:    237.800
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:39      Log-Likelihood: -13778.000
    No. Observations:              1168                 AIC:  27600.000
    Df Residuals:                  1144                 BIC:  27730.000
    Df Model:                        23                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    n
                              SalePrice           R-squared:      0.827
    Dep. Variable:                                                     
    Model:                          OLS      Adj. R-squared:      0.824
    Method:               Least Squares         F-statistic:    248.600
    Date:              Sun, 27 Feb 2022  Prob (F-statistic):      0.000
    Time:                      16:39:41      Log-Likelihood: -13778.000
    No. Observations:              1168                 AIC:  27600.000
    Df Residuals:                  1145                 BIC:  27720.000
    Df Model:                        22                  NaN        NaN
    Covariance Type:          nonrobust                  NaN        NaN
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.827</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.824</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   248.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 27 Feb 2022</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:39:41</td>     <th>  Log-Likelihood:    </th> <td> -13778.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1168</td>      <th>  AIC:               </th> <td>2.760e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1145</td>      <th>  BIC:               </th> <td>2.772e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    22</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 3.961e+04</td> <td> 1.41e+04</td> <td>    2.802</td> <td> 0.005</td> <td> 1.19e+04</td> <td> 6.73e+04</td>
</tr>
<tr>
  <th>x1</th>    <td> -218.8053</td> <td>   27.726</td> <td>   -7.892</td> <td> 0.000</td> <td> -273.204</td> <td> -164.407</td>
</tr>
<tr>
  <th>x2</th>    <td> -206.2838</td> <td>   51.439</td> <td>   -4.010</td> <td> 0.000</td> <td> -307.209</td> <td> -105.359</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.4527</td> <td>    0.099</td> <td>    4.579</td> <td> 0.000</td> <td>    0.259</td> <td>    0.647</td>
</tr>
<tr>
  <th>x4</th>    <td>-1057.0551</td> <td>  718.507</td> <td>   -1.471</td> <td> 0.142</td> <td>-2466.793</td> <td>  352.683</td>
</tr>
<tr>
  <th>x5</th>    <td>  404.6679</td> <td>  166.899</td> <td>    2.425</td> <td> 0.015</td> <td>   77.205</td> <td>  732.131</td>
</tr>
<tr>
  <th>x6</th>    <td> -882.1088</td> <td>  605.298</td> <td>   -1.457</td> <td> 0.145</td> <td>-2069.727</td> <td>  305.510</td>
</tr>
<tr>
  <th>x7</th>    <td> 1.355e+04</td> <td> 1254.086</td> <td>   10.807</td> <td> 0.000</td> <td> 1.11e+04</td> <td>  1.6e+04</td>
</tr>
<tr>
  <th>x8</th>    <td> 5225.6469</td> <td>  976.387</td> <td>    5.352</td> <td> 0.000</td> <td> 3309.938</td> <td> 7141.355</td>
</tr>
<tr>
  <th>x9</th>    <td>  323.8181</td> <td>   55.146</td> <td>    5.872</td> <td> 0.000</td> <td>  215.620</td> <td>  432.017</td>
</tr>
<tr>
  <th>x10</th>   <td> 3529.5065</td> <td> 1188.344</td> <td>    2.970</td> <td> 0.003</td> <td> 1197.931</td> <td> 5861.082</td>
</tr>
<tr>
  <th>x11</th>   <td> -542.3390</td> <td>  311.248</td> <td>   -1.742</td> <td> 0.082</td> <td>-1153.020</td> <td>   68.342</td>
</tr>
<tr>
  <th>x12</th>   <td>-7332.0524</td> <td> 2079.242</td> <td>   -3.526</td> <td> 0.000</td> <td>-1.14e+04</td> <td>-3252.500</td>
</tr>
<tr>
  <th>x13</th>   <td>-6115.3616</td> <td> 1079.651</td> <td>   -5.664</td> <td> 0.000</td> <td>-8233.678</td> <td>-3997.045</td>
</tr>
<tr>
  <th>x14</th>   <td>-4624.8815</td> <td>  941.916</td> <td>   -4.910</td> <td> 0.000</td> <td>-6472.957</td> <td>-2776.806</td>
</tr>
<tr>
  <th>x15</th>   <td>-1150.7877</td> <td>  505.914</td> <td>   -2.275</td> <td> 0.023</td> <td>-2143.410</td> <td> -158.165</td>
</tr>
<tr>
  <th>x16</th>   <td>   53.0107</td> <td>    2.802</td> <td>   18.916</td> <td> 0.000</td> <td>   47.512</td> <td>   58.509</td>
</tr>
<tr>
  <th>x17</th>   <td> 1.001e+04</td> <td> 2206.931</td> <td>    4.534</td> <td> 0.000</td> <td> 5675.287</td> <td> 1.43e+04</td>
</tr>
<tr>
  <th>x18</th>   <td>-9830.6511</td> <td> 1622.746</td> <td>   -6.058</td> <td> 0.000</td> <td> -1.3e+04</td> <td>-6646.762</td>
</tr>
<tr>
  <th>x19</th>   <td> 7980.3117</td> <td> 1767.202</td> <td>    4.516</td> <td> 0.000</td> <td> 4512.995</td> <td> 1.14e+04</td>
</tr>
<tr>
  <th>x20</th>   <td>-1584.7717</td> <td>  862.432</td> <td>   -1.838</td> <td> 0.066</td> <td>-3276.896</td> <td>  107.353</td>
</tr>
<tr>
  <th>x21</th>   <td>-1708.6509</td> <td>  976.316</td> <td>   -1.750</td> <td> 0.080</td> <td>-3624.221</td> <td>  206.919</td>
</tr>
<tr>
  <th>x22</th>   <td>   35.5393</td> <td>    6.218</td> <td>    5.715</td> <td> 0.000</td> <td>   23.339</td> <td>   47.739</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>586.526</td> <th>  Durbin-Watson:     </th> <td>   2.037</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>80151.943</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.275</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>43.503</td>  <th>  Cond. No.          </th> <td>2.27e+05</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.27e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
X_train_copy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>LotShape</th>
      <th>Neighborhood</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>RoofStyle</th>
      <th>Exterior1st</th>
      <th>ExterQual</th>
      <th>BsmtQual</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>KitchenQual</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>254</th>
      <td>20</td>
      <td>70.0</td>
      <td>8400</td>
      <td>3</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>64</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>1314</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>294.0</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>60</td>
      <td>59.0</td>
      <td>7837</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>100</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1571</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>380.0</td>
    </tr>
    <tr>
      <th>638</th>
      <td>30</td>
      <td>67.0</td>
      <td>8777</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>20</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>796</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>799</th>
      <td>50</td>
      <td>60.0</td>
      <td>7200</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>46</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1768</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>380</th>
      <td>50</td>
      <td>50.0</td>
      <td>5000</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>34</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>1691</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>308.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sum_tab = pd.read_html(regressor_OLS.summary().tables[1].as_html(),header=0,index_col=0)[0]
# sum_tab

imp_df = pd.DataFrame({'features':X_train_copy.columns, 'imp':sum_tab.coef[1:]}).sort_values(by=['imp'], ascending=False)

plt.figure(figsize=(5,7))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_285_0.png)
    



```python
imp_df[abs(imp_df.imp)>1700]
imp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x7</th>
      <td>OverallQual</td>
      <td>13550.0000</td>
    </tr>
    <tr>
      <th>x17</th>
      <td>BsmtFullBath</td>
      <td>10010.0000</td>
    </tr>
    <tr>
      <th>x19</th>
      <td>Fireplaces</td>
      <td>7980.3117</td>
    </tr>
    <tr>
      <th>x8</th>
      <td>OverallCond</td>
      <td>5225.6469</td>
    </tr>
    <tr>
      <th>x10</th>
      <td>RoofStyle</td>
      <td>3529.5065</td>
    </tr>
    <tr>
      <th>x5</th>
      <td>Neighborhood</td>
      <td>404.6679</td>
    </tr>
    <tr>
      <th>x9</th>
      <td>YearBuilt</td>
      <td>323.8181</td>
    </tr>
    <tr>
      <th>x16</th>
      <td>GrLivArea</td>
      <td>53.0107</td>
    </tr>
    <tr>
      <th>x22</th>
      <td>GarageArea</td>
      <td>35.5393</td>
    </tr>
    <tr>
      <th>x3</th>
      <td>LotArea</td>
      <td>0.4527</td>
    </tr>
    <tr>
      <th>x2</th>
      <td>LotFrontage</td>
      <td>-206.2838</td>
    </tr>
    <tr>
      <th>x1</th>
      <td>MSSubClass</td>
      <td>-218.8053</td>
    </tr>
    <tr>
      <th>x11</th>
      <td>Exterior1st</td>
      <td>-542.3390</td>
    </tr>
    <tr>
      <th>x6</th>
      <td>HouseStyle</td>
      <td>-882.1088</td>
    </tr>
    <tr>
      <th>x4</th>
      <td>LotShape</td>
      <td>-1057.0551</td>
    </tr>
    <tr>
      <th>x15</th>
      <td>BsmtFinType1</td>
      <td>-1150.7877</td>
    </tr>
    <tr>
      <th>x20</th>
      <td>FireplaceQu</td>
      <td>-1584.7717</td>
    </tr>
    <tr>
      <th>x21</th>
      <td>GarageFinish</td>
      <td>-1708.6509</td>
    </tr>
    <tr>
      <th>x14</th>
      <td>BsmtExposure</td>
      <td>-4624.8815</td>
    </tr>
    <tr>
      <th>x13</th>
      <td>BsmtQual</td>
      <td>-6115.3616</td>
    </tr>
    <tr>
      <th>x12</th>
      <td>ExterQual</td>
      <td>-7332.0524</td>
    </tr>
    <tr>
      <th>x18</th>
      <td>KitchenQual</td>
      <td>-9830.6511</td>
    </tr>
  </tbody>
</table>
</div>




```python
feat_names_stats = X_train_copy.columns
```


```python
X_train_opt = np.append(arr = np.ones((len(X_train_copy),1)).astype(float), 
                        values = X_train_copy[feat_names_stats], axis = 1) 
y_pred = regressor_OLS.predict(X_train_opt.astype(float))

rms_11 = np.sqrt(mean_squared_error(y_train, y_pred))
r2_11 = (r2_score(y_train, y_pred))
print("Train Errors: ",rms, r2)

res_df = pd.DataFrame({'val':y_train, 'pred':y_pred})
sns.scatterplot(res_df.val, res_df.pred)
```

    Train Errors:  25665.740342350215 0.1653565815063145
    




    <AxesSubplot:xlabel='val', ylabel='pred'>




    
![png](output_288_2.png)
    



```python
X_val_opt = np.append(arr = np.ones((len(X_val),1)).astype(float), values = X_val[feat_names_stats], axis = 1) 
y_pred = regressor_OLS.predict(X_val_opt.astype(float))

rms_22 = np.sqrt(mean_squared_error(y_val, y_pred))
r2_22 = (r2_score(y_val, y_pred))
print("Val Errors: ",rms, r2)

res_df = pd.DataFrame({'val':y_val, 'pred':y_pred})
sns.scatterplot(res_df.val, res_df.pred)
```

    Val Errors:  25665.740342350215 0.1653565815063145
    




    <AxesSubplot:xlabel='val', ylabel='pred'>




    
![png](output_289_2.png)
    



```python
model_dict['regressor_OLS'] = [rms_11, rms_22, r2_11, r2_22]
```


```python
regressor_OLS.params.const
```




    39606.10067876608




```python
imp_df = pd.DataFrame({'features':feat_names_stats, 'imp':regressor_OLS.params[1:]}).sort_values(by=['imp'], ascending=False)
# imp_df = imp_df[:12]
plt.figure(figsize=(5,25))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_292_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_293_1.png)
    



```python
sns.scatterplot(y_pred, y_val - y_pred)
```




    <AxesSubplot:ylabel='SalePrice'>




    
![png](output_294_1.png)
    



```python
durbin_watson(y_val - y_pred)

# If this is within the range of 1.5 and 2.5, 
# we will consider there is no autocorrelation
```




    2.033100189244154




```python
plot_acf(y_val - y_pred, alpha =0.05)
plt.show()
```


    
![png](output_296_0.png)
    



```python
plot_pacf(y_val - y_pred, alpha =0.05, lags=50)
plt.show()
```


    
![png](output_297_0.png)
    


So essentially, 
we did not transform the SalePrice with log1p for these results so no inversion is required.

By reducing GrLivArea by one unit we can make the house cheaper by $.
<br> Which means it is a /SQ Ft home.

Going by this logic, GarageArea is /sq-ft.

Also having one less fireplace will help us get the home  cheaper.

#### Verdict

> Backward Feature Elimination using MLExtend:
 
    Imp features: (4 most important)
        
        OverallQual, BsmtQual, GrLivArea, BsmtFullBath
    
> Backward Feature Elimination using P Values:
    It turns out if we keep p-val at 0.05 and don't want R-Sq below 0.75 then we need these features:
    
    ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'LotConfig', 
     'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'HeatingQC', 
     'GrLivArea', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'FireplaceQu', 
     'GarageType', 'GarageFinish', 'GarageArea', 'EnclosedPorch', 'Fence']

## XGBRegressor


```python
X_train_copy = X_train.copy()
```


```python
X_train_copy = X_train_copy.apply(pd.to_numeric)
```

### Baseline


```python
xgbr = xgb.XGBRegressor(seed=27)

xgbr.fit(X_train_copy, y_train)
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=27,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27,
                 subsample=1, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
y_pred = xgbr.predict(X_train_copy)

rms = np.sqrt(mean_squared_error(y_train, y_pred))
r2 = (r2_score(y_train, y_pred))
print(f'Train Errors: RMSE: {rms}, R Sq: {r2}')

pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
y_pred = xgbr.predict(pred_on)

rms = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = (r2_score(y_val, y_pred))
print(f'Val Errors: RMSE: {rms}, R Sq: {r2}')
```

    Train Errors: RMSE: 989.7057444163717, R Sq: 0.9998357768149879
    Val Errors: RMSE: 24744.130491744352, R Sq: 0.8954932148043102
    

### Finding best model


```python
# Params to try

# params = { 'max_depth': [3,6,10],
#            'learning_rate': [0.01, 0.05, 0.1],
#            'n_estimators': [100, 500, 1000],
#            'colsample_bytree': [0.3, 0.7]}

# params = {
#     'max_depth':range(3,10,2),
#     'min_child_weight':[6,8,10,12]
#     'gamma':[i/10.0 for i in range(0,5)]
    
# }

# xgbr = xgb.XGBRegressor(seed = 20)
# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=1)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
```


```python
# Step 1
```


```python
# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# params = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
```


```python
# %%time
# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=2)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
```


```python
# step 2
```


```python
# %%time

# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# params = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[0.1,0.5,1]
# }

# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=2)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
```


```python
# pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = clf.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = (r2_score(y_val, y_pred))
# print("Val Errors: ",rms, r2)

# pred_on = X_test[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = lr.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = (r2_score(y_test, y_pred))
# print(rms, r2)
```


```python
# Step 3
```


```python
# %%time

# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=0.1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# params = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }

# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=2)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

# pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = clf.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = (r2_score(y_val, y_pred))
# print("Val Errors: ",rms, r2)

# pred_on = X_test[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = lr.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = (r2_score(y_test, y_pred))
# print(rms, r2)
```


```python
# Step 4
```


```python
# %%time


# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=0.1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# params = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }

# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=2)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

# pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = clf.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = (r2_score(y_val, y_pred))
# print("Val Errors: ",rms, r2)

# pred_on = X_test[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = lr.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = (r2_score(y_test, y_pred))
# print(rms, r2)
```


```python
# Step 5
```


```python
# %%time

# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=0.1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# params = {
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }

# clf = GridSearchCV(estimator=xgbr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=2)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

# pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = clf.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = (r2_score(y_val, y_pred))
# print("Val Errors: ",rms, r2)

# pred_on = X_test[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = lr.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = (r2_score(y_test, y_pred))
# print(rms, r2)
```

### Final


```python
# xgbr = xgb.XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=0.1,
#  gamma=0,
#  reg_alpha=0.005,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'reg:squarederror',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
```


```python

tr_rmse, val_rmse, r2_1, r2_2 = print_score(xgbr, 
                                            X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric),  y_train, y_val)
```


```python
tr_e = []
val_e = []
max_depths_to_try = [2,3,4,5,6,7,8,9,10]
for i in max_depths_to_try:

    xgbr = xgb.XGBRegressor(
     learning_rate =0.1,
     n_estimators=25,
     max_depth=i,
     min_child_weight=0.1,
     gamma=0,
     reg_alpha=0.005,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'reg:squarederror',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

    xgbr.fit(X_train_copy, y_train)

    tr_rmse, val_rmse, r2_1, r2_2 = print_score(xgbr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric),  y_train, y_val)
    
    tr_e.append(tr_rmse)
    val_e.append(val_rmse)
    
plt.figure(figsize=(15,5))
sns.scatterplot(max_depths_to_try, tr_e, label ='Train RMSE')
sns.scatterplot(max_depths_to_try, val_e, label ='Val RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9dbb4f9a0>




    
![png](output_324_1.png)
    



```python
tr_e = []
val_e = []
n_estimators_to_try = [15, 25, 50, 100, 200, 500, 700, 900, 1000]
for i in max_depths_to_try:

    xgbr = xgb.XGBRegressor(
     learning_rate =0.1,
     n_estimators=i,
     max_depth=5,
     min_child_weight=0.1,
     gamma=0,
     reg_alpha=0.005,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'reg:squarederror',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

    xgbr.fit(X_train_copy, y_train)

    tr_rmse, val_rmse, r2_1, r2_2 = print_score(xgbr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric),  y_train, y_val)
    
    tr_e.append(tr_rmse)
    val_e.append(val_rmse)
    
plt.figure(figsize=(15,5))
sns.scatterplot(n_estimators_to_try, tr_e, label ='Train RMSE')
sns.scatterplot(n_estimators_to_try, val_e, label ='Val RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9e4ec9190>




    
![png](output_325_1.png)
    



```python
xgbr = xgb.XGBRegressor(
     learning_rate =0.1,
     n_estimators=25,
     max_depth=5,
     min_child_weight=0.1,
     gamma=0,
     reg_alpha=0.005,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'reg:squarederror',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

xgbr.fit(X_train_copy, y_train)

tr_rmse, val_rmse, r2_1, r2_2 = print_score(xgbr, X_train_copy, 
                                            X_val[X_train_copy.columns].apply(pd.to_numeric), y_train, y_val)
tr_rmse, val_rmse, r2_1, r2_2
```




    (24596.82421365921, 25259.019270890327, 0.89856674470391, 0.891098700493482)




```python
model_dict['xgbr'] = [tr_rmse, val_rmse, r2_1, r2_2]
tr_rmse, val_rmse, r2_1, r2_2 
```




    (24596.82421365921, 25259.019270890327, 0.89856674470391, 0.891098700493482)




```python
imp_df = pd.DataFrame({'features':X_train_copy.columns, 'imp':xgbr.feature_importances_}).sort_values(by=['imp'], ascending=False)
imp_df = imp_df[:20]
plt.figure(figsize=(5,8))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_328_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_329_1.png)
    


## Random Forest Regressor

### Baseline


```python
rfr = RandomForestRegressor()
rfr.fit(X_train_copy, y_train)

print_score(rfr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric), y_train, y_val)
```




    (12019.941095892682, 22176.28440119547, 0.9757770536336238, 0.9160583196117609)



### Finding the best model


```python
# %%time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

# params={  'max_depth': [5, 10, None], 
#          'max_features': ['auto'], 
#          'n_estimators': [10, 100, 250, 500, 700, 800, 1000]}

# rfr = RandomForestRegressor()
# clf = GridSearchCV(estimator=rfr, 
#                    param_grid=params,
#                    scoring='neg_mean_squared_error', 
#                    verbose=1)
# clf.fit(X_train_copy, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

# pred_on = X_val[X_train_copy.columns].apply(pd.to_numeric)
# y_pred = clf.predict(pred_on)

# rms = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = (r2_score(y_val, y_pred))
# print("Val Errors: ",rms, r2)
```


```python
# Fitting 5 folds for each of 36 candidates, totalling 180 fits
# Best parameters: {'bootstrap': True, 'max_depth': None, 'max_features': 'log2', 'n_estimators': 500}
# Lowest RMSE:  31122.676660515
# Val Errors:  23372.077786452188 0.9067616155645672
# 41278.5966307294 0.8189044399992987
# Wall time: 7min 54s
```

### Final


```python
tr_e = []
val_e = []
max_depths_to_try = [2,3,4,5,6,7,8,9,10]

for i in max_depths_to_try:

    rfr = RandomForestRegressor(bootstrap= True, max_depth=i, max_features= 'log2', n_estimators=25)
    rfr.fit(X_train_copy, y_train)

    tr_rmse, val_rmse, r2_1, r2_2 = print_score(rfr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric), y_train, y_val)
    
    tr_e.append(tr_rmse)
    val_e.append(val_rmse)
    
plt.figure(figsize=(15,5))
sns.scatterplot(max_depths_to_try, tr_e, label ='Train RMSE')
sns.scatterplot(max_depths_to_try, val_e, label ='Val RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9e66a6bb0>




    
![png](output_337_1.png)
    



```python
tr_e = []
val_e = []
n_estimators_to_try = [15, 25, 50, 100, 250, 500, 700, 900, 1000]

for i in n_estimators_to_try:

    rfr = RandomForestRegressor(bootstrap= True, max_depth=5, max_features= 'log2', n_estimators=i)
    rfr.fit(X_train_copy, y_train)

    tr_rmse, val_rmse, r2_1, r2_2 = print_score(rfr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric), y_train, y_val)
    
    tr_e.append(tr_rmse)
    val_e.append(val_rmse)
    
plt.figure(figsize=(15,5))
sns.scatterplot(n_estimators_to_try, tr_e, label ='Train RMSE')
sns.scatterplot(n_estimators_to_try, val_e, label ='Val RMSE')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a9e501e4f0>




    
![png](output_338_1.png)
    



```python
rfr = RandomForestRegressor(bootstrap= True, max_depth=5, max_features= 'log2', n_estimators=500)
rfr.fit(X_train_copy, y_train)

tr_rmse, val_rmse, r2_1, r2_2  = print_score(rfr, X_train_copy, X_val[X_train_copy.columns].apply(pd.to_numeric), y_train, y_val)
tr_rmse, val_rmse, r2_1, r2_2 
```




    (28498.145353667478,
     28572.109250289664,
     0.8638381375143974,
     0.8606571471719313)




```python
model_dict['rfr'] = [tr_rmse, val_rmse, r2_1, r2_2]
tr_rmse, val_rmse, r2_1, r2_2 
```




    (28498.145353667478,
     28572.109250289664,
     0.8638381375143974,
     0.8606571471719313)




```python
imp_df = pd.DataFrame({'features':X_train_copy.columns, 'imp':rfr.feature_importances_}).sort_values(by=['imp'], ascending=False)
imp_df = imp_df[:20]
plt.figure(figsize=(5,8))
sns.barplot(x=imp_df.imp, y=imp_df.features)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_341_0.png)
    



```python
sns.histplot(y_val - y_pred)
```




    <AxesSubplot:xlabel='SalePrice', ylabel='Count'>




    
![png](output_342_1.png)
    


## Comparing Models


```python
# vanilla_linear_regression
# ENreg
# lasso
# ridge
# feat_names, lr_mlxtend
# regressor_OLS
# xgbr
# rfr
```


```python
model_dict
```




    {'vanilla_linear_regression': [32053.095397745885,
      25779.714268262076,
      0.8277488060019488,
      0.8865625924712075],
     'ENreg': [38316.81050445832,
      32552.688825270074,
      0.7538493476031198,
      0.819126964499927],
     'lasso': [32053.095747705964,
      25778.509259490234,
      0.8277488022406245,
      0.8865731969236088],
     'ridge': [32296.476830362637,
      25665.740342350215,
      0.8251230437768077,
      0.8875634047180159],
     'lr_mlxtend': [32407.801010332543,
      25693.641780058737,
      0.8239153834818104,
      0.8873188103424505],
     'regressor_OLS': [32134.547039115896,
      26077.941891246235,
      0.8268722623399343,
      0.8839228543483857],
     'xgbr': [24596.82421365921,
      25259.019270890327,
      0.89856674470391,
      0.891098700493482],
     'rfr': [28498.145353667478,
      28572.109250289664,
      0.8638381375143974,
      0.8606571471719313]}




```python
model_dict.keys()
```




    dict_keys(['vanilla_linear_regression', 'ENreg', 'lasso', 'ridge', 'lr_mlxtend', 'regressor_OLS', 'xgbr', 'rfr'])




```python
for i in model_dict.keys():
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    sns.barplot(['train_rmse', 'val_rmse'], model_dict[i][:2])
    plt.subplot(1,2,2)
    sns.barplot(['train_r2', 'val_r2'], model_dict[i][2:])
    plt.suptitle(i)
    plt.show()
```


    
![png](output_347_0.png)
    



    
![png](output_347_1.png)
    



    
![png](output_347_2.png)
    



    
![png](output_347_3.png)
    



    
![png](output_347_4.png)
    



    
![png](output_347_5.png)
    



    
![png](output_347_6.png)
    



    
![png](output_347_7.png)
    



```python
train_rmse = []
val_rmse = []

train_r2 = []
val_r2 = []

for i in model_dict.keys():
    train_rmse.append(model_dict[i][0])
    val_rmse.append(model_dict[i][1])
    
    train_r2.append(model_dict[i][2])
    val_r2.append(model_dict[i][3])

plt.figure(figsize=(15,5))
    
plt.subplot(1,2,1)
plt.title('Train RMSE')
sns.barplot(list(model_dict.keys()), train_rmse)
plt.xticks(rotation=85)

plt.subplot(1,2,2)
plt.title('Val RMSE')
sns.barplot(list(model_dict.keys()), val_rmse)
plt.xticks(rotation=85)

plt.show()

plt.figure(figsize=(15,5))
    
plt.subplot(1,2,1)
plt.title('Train R2')
sns.barplot(list(model_dict.keys()), train_r2)
plt.xticks(rotation=85)

plt.subplot(1,2,2)
plt.title('Val R2')
sns.barplot(list(model_dict.keys()), val_r2)
plt.xticks(rotation=85)

plt.show()
```


    
![png](output_348_0.png)
    



    
![png](output_348_1.png)
    



```python
# vanilla
# xgbr
```


```python
model_dict.keys()
```




    dict_keys(['vanilla_linear_regression', 'ENreg', 'lasso', 'ridge', 'lr_mlxtend', 'regressor_OLS', 'xgbr', 'rfr'])




```python
def test_score(name, model, X_test, perform_df):
    y_pred = model.predict(X_test)

    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = (r2_score(y_test, y_pred))
#     print(f'{name} Test Errors: RMSE: {rms}, R Sq: {r2}')
    perform_df = perform_df.append({'Model':name, 'Test RMSE':rms, 'Test R2':r2}, ignore_index=True)
    return rms, r2, perform_df
```


```python
perform_df = pd.DataFrame({'Model':[], 'Test RMSE':[], 'Test R2':[]})

rms, r2, perform_df = test_score('Vanilla Lin Reg', vanilla_linear_regression, X_test, perform_df)
rms, r2, perform_df = test_score('Lasso', lasso, X_test[X_train.columns], perform_df)
rms, r2, perform_df = test_score('Rigde', ridge, X_test[X_train.columns], perform_df)
rms, r2, perform_df = test_score('Elastic Net', ENreg, X_test[X_train.columns], perform_df)
rms, r2, perform_df = test_score('mlxtend_lin', lr_mlxtend, X_test[feat_names], perform_df)

X_test_opt = np.append(arr = np.ones((len(X_test),1)).astype(float), 
                        values = X_test[feat_names_stats], axis = 1)
rms, r2, perform_df = test_score('statsmodel Lin Reg', regressor_OLS, X_test_opt, perform_df)

rms, r2, perform_df = test_score('XGBoost', xgbr, X_test[X_train_copy.columns].apply(pd.to_numeric), perform_df)
rms, r2, perform_df = test_score('Random Forest', rfr, X_test[X_train_copy.columns].apply(pd.to_numeric), perform_df)

perform_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Test RMSE</th>
      <th>Test R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Vanilla Lin Reg</td>
      <td>43736.575308</td>
      <td>0.796695</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lasso</td>
      <td>43736.007030</td>
      <td>0.796701</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rigde</td>
      <td>44458.340109</td>
      <td>0.789930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elastic Net</td>
      <td>46706.521566</td>
      <td>0.768147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mlxtend_lin</td>
      <td>43237.904333</td>
      <td>0.801305</td>
    </tr>
    <tr>
      <th>5</th>
      <td>statsmodel Lin Reg</td>
      <td>43628.545621</td>
      <td>0.797698</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost</td>
      <td>49059.087245</td>
      <td>0.744202</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Random Forest</td>
      <td>48187.309314</td>
      <td>0.753212</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Test RMSE')
plt.xticks(rotation=85)
sns.barplot(data=perform_df, x='Model', y='Test RMSE')

plt.subplot(1,2,2)
plt.title('Test R2')
sns.barplot(data=perform_df, x='Model', y='Test R2')
plt.xticks(rotation=85)

plt.show()
```


    
![png](output_353_0.png)
    


The best model is:


```python
perform_df[perform_df.Model == 'mlxtend_lin']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Test RMSE</th>
      <th>Test R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>mlxtend_lin</td>
      <td>43237.904333</td>
      <td>0.801305</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred = lr_mlxtend.predict(X_test[feat_names])
```


```python
plt.figure(figsize=(15,5))
# plt.bar(x = np.arange(len(y_test)), y = y_test, height=10)
# plt.bar(x = np.arange(len(y_test)), y = y_pred, height=10)

plt.bar(np.arange(len(y_test)), y_test, color='r', label='True')
plt.bar(np.arange(len(y_test)), y_pred, bottom=y_test, color='b', label='Predicted')
plt.legend()
plt.show()
```


    
![png](output_357_0.png)
    


## Predicting on given Test Data


```python
pred_on = test[feat_names]
```


```python
X_train[feat_names].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 254 to 1126
    Data columns (total 20 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   MSSubClass    1168 non-null   object 
     1   LotArea       1168 non-null   int64  
     2   LotShape      1168 non-null   int32  
     3   Neighborhood  1168 non-null   int32  
     4   HouseStyle    1168 non-null   int32  
     5   OverallQual   1168 non-null   object 
     6   OverallCond   1168 non-null   object 
     7   YearBuilt     1168 non-null   int32  
     8   RoofStyle     1168 non-null   int32  
     9   ExterQual     1168 non-null   int32  
     10  BsmtQual      1168 non-null   int32  
     11  BsmtExposure  1168 non-null   int32  
     12  BsmtFinType1  1168 non-null   int32  
     13  GrLivArea     1168 non-null   int64  
     14  BsmtFullBath  1168 non-null   object 
     15  KitchenQual   1168 non-null   int32  
     16  Fireplaces    1168 non-null   object 
     17  FireplaceQu   1168 non-null   int32  
     18  GarageFinish  1168 non-null   int32  
     19  GarageArea    1168 non-null   float64
    dtypes: float64(1), int32(12), int64(2), object(5)
    memory usage: 136.9+ KB
    


```python
pred_on.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1459 entries, 0 to 1458
    Data columns (total 20 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   MSSubClass    1459 non-null   object 
     1   LotArea       1459 non-null   int64  
     2   LotShape      1459 non-null   int32  
     3   Neighborhood  1459 non-null   int32  
     4   HouseStyle    1459 non-null   int32  
     5   OverallQual   1459 non-null   object 
     6   OverallCond   1459 non-null   object 
     7   YearBuilt     1459 non-null   int32  
     8   RoofStyle     1459 non-null   int32  
     9   ExterQual     1459 non-null   int32  
     10  BsmtQual      1459 non-null   int32  
     11  BsmtExposure  1459 non-null   int32  
     12  BsmtFinType1  1459 non-null   int32  
     13  GrLivArea     1459 non-null   int64  
     14  BsmtFullBath  1457 non-null   object 
     15  KitchenQual   1459 non-null   int32  
     16  Fireplaces    1459 non-null   object 
     17  FireplaceQu   1459 non-null   int32  
     18  GarageFinish  1459 non-null   int32  
     19  GarageArea    1458 non-null   float64
    dtypes: float64(1), int32(12), int64(2), object(5)
    memory usage: 171.0+ KB
    


```python
X_train.GarageArea.median()
```




    482.0




```python
pred_on.BsmtFullBath.unique()
```




    array([0.0, 1.0, 2.0, 3.0, nan], dtype=object)




```python
pred_on.loc[pred_on['BsmtFullBath'].isnull(), 'BsmtFullBath'] = '0'
pred_on.loc[pred_on['GarageArea'].isnull(), 'GarageArea'] = pred_on['GarageArea'].median()
```


```python
final_submission = lr_mlxtend.predict(pred_on)
```


```python
np.mean(final_submission), np.median(final_submission)
```




    (174945.87031742427, 160487.98266231493)




```python
plt.figure(figsize=(15,5))
sns.barplot(x=np.arange(len(pred_on)), y=final_submission)
```




    <AxesSubplot:>




    
![png](output_367_1.png)
    



```python
submission_file = pd.DataFrame({'Id':pd.read_csv('test.csv').Id, 'SalePrice':final_submission})
```


```python
submission_file.to_csv('Submission.csv',index=False)
```


```python
pd.read_csv('Submission.csv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>108576.031452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>156325.529086</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>166310.895958</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>184840.397003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>190607.283420</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>53882.327255</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>55481.742287</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>153527.893569</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>103194.848470</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>224412.532502</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 2 columns</p>
</div>



## Improvements Possible:


```python
# - Treating the outliers with more subject matter expertise.
# - New features can be added through feature engineering.
# - Ensemble can be tried - like stacking regressor
# - Can try with log1p transformation for Skewness
# - Can have more elaborate interpretation of feature importances and coefficients for better business use.
```
