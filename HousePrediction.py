import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 10)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st


Data_frame = pd.read_csv("Bengaluru_House_Data.csv", na_values=[' ?'])
Data_frame_copy = Data_frame.copy()
Data_frame.drop(['availability', 'society'], axis=1, inplace=True)
Data_frame.dropna(subset = ['location', 'size', 'bath'], inplace=True)
Data_frame['balcony'].replace(np.nan, Data_frame['balcony'].mean(), inplace=True)
Data_frame['BHK'] = Data_frame['size'].apply(lambda x: int(x.split(' ')[0]))
Data_frame.drop(['size'], axis=1, inplace=True)
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

~Data_frame['total_sqft'].apply(is_float)
Temp_df = Data_frame[~Data_frame['total_sqft'].apply(is_float)]
def abnormal_change(x):
    if "Sq. Meter" in x:
        num = x.split("S")
        result = float(num[0]) * 10.76
        return result
    if "Perch" in x:
        num = x.split("P")
        result = float(num[0]) * 272.25
        return result
    if "Sq. Yards" in x:
        num = x.split("S")
        result = float(num[0]) * 9.00
        return result
    if "Acres" in x:
        num = x.split("A")
        result = float(num[0]) * 43560.04
        return result
    if "Cents" in x:
        num = x.split("C")
        result = float(num[0]) * 435.56
        return result
    if "Guntha" in x:
        num = x.split("G")
        result = float(num[0]) * 1089.00
        return result
    if "Grounds" in x:
        num = x.split("G")
        result = float(num[0]) * 2400.35
        return result

def convert_range_to_num(x):
    if '-' in x:
        num = x.split('-')
        result1 = (float(num[0]) + float(num[1]))/2
    else:
        bool_val = x.upper().isupper()
        if bool_val == True:
            result1 = abnormal_change(x)
        else:
            result1 = float(x)
    return result1

Data_frame['total_sqft'] = Data_frame['total_sqft'].apply(convert_range_to_num)

Data_frame.rename(columns = {'bath':'bathroom'}, inplace=True)
Data_frame['price_per_sqft'] = Data_frame['price']*100000 / Data_frame['total_sqft']
count_location = len(Data_frame['location'].unique())
pd.set_option('display.max_rows', count_location)
Location_data = Data_frame.groupby('location')['location'].agg('count').sort_values(ascending=False)
filt = Location_data <= 15
Location_data_less_than_15 = Location_data[filt]
Data_frame['location'] = Data_frame['location'].apply(lambda x: 'Other' if x in Location_data_less_than_15 else x)
count_location_dimen_red = len(Data_frame['location'].unique())

# Applying Quantile Based Flooring and capping
lower_bound = Data_frame['total_sqft'].quantile(0.10)
upper_bound = Data_frame['total_sqft'].quantile(0.90)
Data_frame['total_sqft'] = np.where(Data_frame['total_sqft'] < lower_bound, lower_bound, Data_frame['total_sqft'])
Data_frame['total_sqft'] = np.where(Data_frame['total_sqft'] > upper_bound, upper_bound, Data_frame['total_sqft'])

# Bathroom - small quanitites of Outliers so Replace them with median 
median = Data_frame['bathroom'].quantile(0.50)
upper_out = Data_frame['bathroom'].quantile(0.95)
Data_frame['bathroom'] = np.where(Data_frame['bathroom'] > upper_out, median, Data_frame['bathroom'])

# Applying Quantile Based Flooring and capping
lower_bound = Data_frame['price'].quantile(0.10)
upper_bound = Data_frame['price'].quantile(0.90)
Data_frame['price'] = np.where(Data_frame['price'] < lower_bound, lower_bound, Data_frame['price'])
Data_frame['price'] = np.where(Data_frame['price'] > upper_bound, upper_bound, Data_frame['price'])

# BHK - small quanitites of Outliers so Replace them with median 
median = Data_frame['BHK'].quantile(0.50)
upper_out = Data_frame['BHK'].quantile(0.98)
Data_frame['BHK'] = np.where(Data_frame['BHK'] > upper_out, median, Data_frame['BHK'])

Data_frame['balcony'] = Data_frame['balcony'].astype('int')

# Applying Quantile Based Flooring and capping
lower_bound = Data_frame['price_per_sqft'].quantile(0.10)
upper_bound = Data_frame['price_per_sqft'].quantile(0.90)
Data_frame['price_per_sqft'] = np.where(Data_frame['price_per_sqft'] < lower_bound, lower_bound, Data_frame['price_per_sqft'])
Data_frame['price_per_sqft'] = np.where(Data_frame['price_per_sqft'] > upper_bound, upper_bound, Data_frame['price_per_sqft'])

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis=0)
Data_frame = remove_bhk_outliers(Data_frame)
Data_frame.drop(['price_per_sqft'], axis=1, inplace=True)

Data_frame.to_csv("Cleaned_data.csv")
one_dum = pd.get_dummies(Data_frame['area_type'])
Data_frame = pd.concat([Data_frame, one_dum], axis=1)
Data_frame.drop(['area_type'], axis=1, inplace=True)
ne_dum = pd.get_dummies(Data_frame['location'])
Data_frame = pd.concat([Data_frame, ne_dum], axis=1)
Data_frame.drop(['location'], axis=1, inplace=True)
X = Data_frame.drop(['price'], axis=1)
Y = Data_frame['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.17, random_state=42)

LinearModel = LinearRegression()
LinearModel.fit(X_train, Y_train)
Y_pred = LinearModel.predict(X_test)

if "Y_pred" not in st.session_state:
    st.session_state['Y_pred']=Y_pred
if "Y_test" not in st.session_state:
    st.session_state['Y_test']=Y_test

cv = KFold(n_splits=8, random_state=None)
cross_val_score(LinearRegression(), X, Y, cv=cv)
result = cross_val_score(LinearRegression(), X, Y, cv=cv)
print(f"Average Accuracy - {result.mean()}")

import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as obj:
    pickle.dump(LinearModel, obj)

import json
columns = {
    'Columns': [col.lower() for col in X.columns]
}
with open("Columns.json", 'w') as f:
    f.write(json.dumps(columns))
