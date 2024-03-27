import numpy as np
import pandas as pd
import pickle
import streamlit as st
import json
import math
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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

# ~Data_frame['total_sqft'].apply(is_float)
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
result = None
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
# print(f"Average Accuracy - {result.mean()}")

import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as obj:
    pickle.dump(LinearModel, obj)

import json
columns = {
    'Columns': [col.lower() for col in X.columns]
}
with open("Columns.json", 'w') as f:
    f.write(json.dumps(columns))

with open(
        r"bangalore_home_prices_model.pickle", 'rb') as f:
    __model = pickle.load(f)

with open(r"Columns.json", 'r') as obj:
    __data_columns = json.load(obj)["Columns"]
    __area_types = __data_columns[4:8]
    __locations = __data_columns[8:]


def get_predicted_price(area_type, location, sqft, balcony, bathroom, BHK):
    try:
        area_index = __data_columns.index(area_type.lower())
        loc_index = __data_columns.index(location.lower())
    except ValueError as e:
        area_index = -1
        loc_index = -1

    lis = np.zeros(len(__data_columns))
    lis[0] = sqft
    lis[1] = bathroom
    lis[2] = balcony
    lis[3] = BHK

    if loc_index >= 0 and area_index >= 0:
        lis[area_index] = 1
        lis[loc_index] = 1

    price = round(__model.predict([lis])[0], 2)
    strp = ' lakhs'

    if math.log10(price) >= 2:
        price = price / 100
        price = round(price, 2)
        strp = " crores"

    return str(price) + strp


# def main():
#     global result
#     st.title("House Price Predictor")
#     html_temp = """
#            <div>
#            <h2>House Price Prediction ML app</h2>
#            </div>
#            """
#     st.markdown(html_temp, unsafe_allow_html=True)
#     total_sqft = st.text_input("Total_sqft")
#     balcony = st.text_input("Number of Balconies")
#     bathroom = st.text_input("Number of Bathrooms")
#     BHK = st.text_input("BHK")
#     area_type = st.selectbox("Area Type", __area_types)
#     location = st.selectbox("Location", __locations)

#     if st.button("Predict"):
#         result = get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)
#         st.balloons()
#     st.success(f"Price = {result}")


# if __name__ == "__main__":
#     main()

def main():
    global result
    st.title("House Price Predictor")
    st.write(
        "Welcome to the House Price Predictor! Enter the details below to get an estimate of the house price."
    )

    # Add a brief description or instruction for the user
    st.write(
        "Please input the details such as total square feet, number of balconies, number of bathrooms, and BHK (Bedrooms, Hall, Kitchen)."
    )

    # Input fields with descriptions
    total_sqft = st.text_input("Total Square Feet", help="Enter the total area of the house in square feet.")
    balcony = st.text_input("Number of Balconies", help="Specify the number of balconies in the house.")
    bathroom = st.text_input("Number of Bathrooms", help="Enter the total number of bathrooms in the house.")
    BHK = st.text_input("BHK (Bedrooms, Hall, Kitchen)", help="Specify the number of bedrooms, hall, and kitchen in the house.")

    # Dropdowns for categorical features
    area_type = st.selectbox("Area Type", __area_types, help="Select the type of area.")
    location = st.selectbox("Location", __locations, help="Select the location of the house.")

    # Button to trigger prediction
    if st.button("Predict"):
        result = get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)
        st.balloons()

    # Display the predicted price
    st.success(f"Estimated House Price: {result}")

    # Optionally, you can add more information or insights about the prediction
    st.write(
        "This prediction is based on a machine learning model trained on Bengaluru house data. The model takes into account various features to provide an estimate."
    )

    # Add a footer with any additional information or credits
    st.write("Powered by House Price Prediction Model. © 2023")

if __name__ == "__main__":
    main()

