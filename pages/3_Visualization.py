import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import folium_static  # Importing folium_static for Streamlit integration


# Data_frame=pd.read_csv("Cleaned_data.csv")

Data_frame = pd.read_csv("Cleaned_data.csv")
Data_frame['availability'] = np.random.choice(['Available', 'Not Available'], size=len(Data_frame))

# # Welcome Page
st.title("Welcome to the House Price Analysis App")

# Sidebar Navigation
page = st.sidebar.selectbox("Select a page", ["Scatter Chart Analysis","Balcony Analysis", "Total Square Feet Analysis", "Bathroom Count Analysis",
                                              "Price Analysis", "BHK Analysis", "Scatter Chart Analysis",
                                              "Bar Plot Analysis", "Heatmap Analysis", "Actual vs Predicted Value",
                                              "Regression Plot"])

# # Balcony Analysis


# # Scatter Chart
# if page == "Scatter Chart Analysis":
#     st.title("Scatter Chart Analysis")
#     selected_location = st.sidebar.selectbox("Select Location", Data_frame['location'].unique())
#     bhk2 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 2)]
#     bhk3 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 3)]

#     # Create a folium map centered around Bengaluru
#     scatter_map = folium.Map(location=[12.971598, 77.594562], zoom_start=12)

#     # Add markers for each property location
#     for index, property_data in bhk2.iterrows():
#         folium.Marker(
#             location=[property_data['lat'], property_data['lon']],
#             popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
#             icon=folium.Icon(color='blue')
#         ).add_to(scatter_map)

#     for index, property_data in bhk3.iterrows():
#         folium.Marker(
#             location=[property_data['lat'], property_data['lon']],
#             popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
#             icon=folium.Icon(color='green', icon='plus')
#         ).add_to(scatter_map)

#     # Display the map
#     st.title(f"Scatter Chart for {selected_location} with Property Locations")
#     folium_static(scatter_map)


# elif page == "Balcony Analysis":
#     st.title("Balcony Analysis")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='balcony', data=Data_frame, ax=ax)
#     plt.title("Balcony BoxPlot")
#     st.pyplot(fig)

# # Total Square Feet Analysis
# elif page == "Total Square Feet Analysis":
#     st.title("Total Square Feet Analysis")
#     st.subheader("Histogram for Total Square Feet")
#     fig, ax = plt.subplots()
#     sns.histplot(x='total_sqft', data=Data_frame, bins=30, kde=True, ax=ax)
#     plt.xlim([0, 6000])
#     plt.title("Total Square Feet Distribution")
#     st.pyplot(fig)

#     st.subheader("Boxplot for Total Square Feet")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='total_sqft', data=Data_frame, ax=ax)
#     plt.title("Total Square Feet Boxplot")
#     st.pyplot(fig)

# # Bathroom Count Analysis
# elif page == "Bathroom Count Analysis":
#     st.title("Bathroom Count Analysis")
#     st.subheader("Boxplot for Bathroom Count")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='bathroom', data=Data_frame, ax=ax)
#     plt.title("Bathroom Count Boxplot")
#     st.pyplot(fig)

#     st.subheader("Histogram for Bathroom Count")
#     fig, ax = plt.subplots()
#     plt.hist(Data_frame['bathroom'], bins=20, color='skyblue', edgecolor='black')
#     plt.xlabel("Number of Bathrooms")
#     plt.ylabel("Count")
#     plt.title("Bathroom Count Distribution")
#     st.pyplot(fig)

# # Price Analysis
# elif page == "Price Analysis":
#     st.title("Price Analysis")
#     st.subheader("Histogram for Price")
#     fig, ax = plt.subplots()
#     sns.histplot(x='price', data=Data_frame, bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black')
#     plt.xlim([0, 2000])
#     plt.xlabel("Price")
#     plt.ylabel("Count")
#     plt.title("Price Distribution")
#     st.pyplot(fig)

#     st.subheader("Boxplot for Price")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='price', data=Data_frame, ax=ax)
#     plt.title("Price Boxplot")
#     st.pyplot(fig)

# # BHK Analysis
# elif page == "BHK Analysis":
#     st.title("BHK Analysis")
#     st.subheader("Histogram for BHK")
#     fig, ax = plt.subplots()
#     ax.hist(Data_frame['BHK'], bins=20, color='skyblue', edgecolor='black')
#     plt.xlabel("BHK")
#     plt.ylabel("Count")
#     plt.title("BHK Distribution")
#     st.pyplot(fig)

#     st.subheader("Boxplot for Number of Bedrooms")
#     fig, ax = plt.subplots()
#     sns.boxplot(x='BHK', data=Data_frame, ax=ax)
#     plt.title("Number of Bedrooms Boxplot")
#     st.pyplot(fig)

# # Scatter Chart
# elif page == "Scatter Chart Analysis":
#     st.title("Scatter Chart Analysis")
#     selected_location = st.sidebar.selectbox("Select Location", Data_frame['location'].unique())
#     bhk2 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 2)]
#     bhk3 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 3)]

#     fig, ax = plt.subplots()
#     ax.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
#     ax.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
#     plt.xlabel("Total Square Feet Area")
#     plt.ylabel("Price (Lakh Indian Rupees)")
#     plt.title(f"Scatter Chart for {selected_location}")
#     plt.legend()
#     st.pyplot(fig)

# # Bar Plot - Bathroom and BHK
# elif page == "Bar Plot Analysis":
#     st.title('Bar Plot - Bathroom and BHK')
#     fig, ax = plt.subplots()
#     sns.barplot(x='BHK', y='bathroom', data=Data_frame, palette='Blues', ax=ax)
#     plt.title('Bar Plot - Bathroom and BHK')
#     plt.xlabel('BHK')
#     plt.ylabel('Bathroom')
#     st.pyplot(fig)

# # Heatmap
# elif page == "Heatmap Analysis":
#     st.title('Heatmap Analysis')
#     corr_matrix = Data_frame.corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
#     st.pyplot(fig)

# # Actual vs Predicted Value
# elif page == "Actual vs Predicted Value":
#     st.title('Actual vs Predicted Value')
#     fig, ax = plt.subplots()
#     axis = sns.distplot(x=st.session_state['Y_test'], hist=False, color='red', label='Actual Data')
#     sns.distplot(x=st.session_state['Y_pred'], hist=False, color='blue', label='Predicted Data', ax=axis)
#     plt.xlabel('Price')
#     plt.title('Actual Value vs Predicted Value')
#     plt.legend(loc='best')
#     st.pyplot(fig)

# # Regression Plot
# elif page == "Regression Plot":
#     st.title('Regression Plot - Actual vs Predicted')
#     sns.set(color_codes=True)
#     sns.set_style("white")
#     fig, ax = plt.subplots()
#     ax = sns.regplot(x=st.session_state['Y_test'], y=st.session_state['Y_pred'], scatter_kws={'alpha': 0.4})
#     ax.set_xlabel('Original Value - Price', fontsize='large', fontweight='bold')
#     ax.set_ylabel('Predicted Value - Price', fontsize='large', fontweight='bold')
#     ax.set_xlim(35, 135)
#     ax.set_ylim(15, 135)
#     st.pyplot(fig)
# ========================================================================================================

import folium
from streamlit_folium import folium_static  # Importing folium_static for Streamlit integration
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter Chart
# Scatter Chart with Geoplot
# if page == "Scatter Chart Analysis":
#     st.title("Scatter Chart Analysis")
#     selected_location = st.sidebar.selectbox("Select Location", Data_frame['location'].unique())
#     bhk2 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 2)]
#     bhk3 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 3)]

#     # Create a folium map centered around Bengaluru
#     scatter_map = folium.Map(location=[12.971598, 77.594562], zoom_start=12)

#     # Add markers for each property location
#     for index, property_data in bhk2.iterrows():
#         folium.Marker(
#             location=[property_data['latitude'], property_data['longitude']],  # Replace with actual column names
#             popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
#             icon=folium.Icon(color='blue')
#         ).add_to(scatter_map)

#     for index, property_data in bhk3.iterrows():
#         folium.Marker(
#             location=[property_data['latitude'], property_data['longitude']],  # Replace with actual column names
#             popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
#             icon=folium.Icon(color='green', icon='plus')
#         ).add_to(scatter_map)

#     # Display the map
#     st.title(f"Scatter Chart for {selected_location} with Property Locations")
#     folium_static(scatter_map)

from folium.plugins import MarkerCluster

# Scatter Chart with MarkerCluster for Available Houses
if page == "Scatter Chart Analysis":
    st.title("Scatter Chart Analysis")
    selected_location = st.sidebar.selectbox("Select Location", Data_frame['location'].unique())
    bhk2 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 2)]
    bhk3 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 3)]

    # Create a folium map centered around Bengaluru
    scatter_map = folium.Map(location=[12.971598, 77.594562], zoom_start=12)

    # Create a MarkerCluster for better visualization
    marker_cluster = MarkerCluster().add_to(scatter_map)

    # Add markers for each property location with custom icons
    for index, property_data in bhk2.iterrows():
        icon_color = 'green' if property_data['availability'] == 'Available' else 'red'
        folium.Marker(
            location=[property_data['total_sqft'] * 0.001, property_data['price'] * 0.001],
            popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
            icon=folium.Icon(color=icon_color)
        ).add_to(marker_cluster)

    for index, property_data in bhk3.iterrows():
        icon_color = 'green' if property_data['availability'] == 'Available' else 'red'
        folium.Marker(
            location=[property_data['total_sqft'] * 0.001, property_data['price'] * 0.001],
            popup=f"Price: {property_data['price']} INR\nBHK: {property_data['BHK']}",
            icon=folium.Icon(color=icon_color)
        ).add_to(marker_cluster)

    # Display the map
    st.title(f"Scatter Chart for {selected_location} with Available Houses Highlighted")
    folium_static(scatter_map)


# Balcony Analysis
elif page == "Balcony Analysis":
    st.title("Balcony Analysis")
    fig, ax = plt.subplots()
    sns.boxplot(x='balcony', data=Data_frame, ax=ax)
    plt.title("Balcony BoxPlot")
    st.pyplot(fig)

# Total Square Feet Analysis
elif page == "Total Square Feet Analysis":
    st.title("Total Square Feet Analysis")
    st.subheader("Histogram for Total Square Feet")
    fig, ax = plt.subplots()
    sns.histplot(x='total_sqft', data=Data_frame, bins=30, kde=True, ax=ax)
    plt.xlim([0, 6000])
    plt.title("Total Square Feet Distribution")
    st.pyplot(fig)

    st.subheader("Boxplot for Total Square Feet")
    fig, ax = plt.subplots()
    sns.boxplot(x='total_sqft', data=Data_frame, ax=ax)
    plt.title("Total Square Feet Boxplot")
    st.pyplot(fig)

# Bathroom Count Analysis
elif page == "Bathroom Count Analysis":
    st.title("Bathroom Count Analysis")
    st.subheader("Boxplot for Bathroom Count")
    fig, ax = plt.subplots()
    sns.boxplot(x='bathroom', data=Data_frame, ax=ax)
    plt.title("Bathroom Count Boxplot")
    st.pyplot(fig)

    st.subheader("Histogram for Bathroom Count")
    fig, ax = plt.subplots()
    plt.hist(Data_frame['bathroom'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Count")
    plt.title("Bathroom Count Distribution")
    st.pyplot(fig)

# Price Analysis
elif page == "Price Analysis":
    st.title("Price Analysis")
    st.subheader("Histogram for Price")
    fig, ax = plt.subplots()
    sns.histplot(x='price', data=Data_frame, bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black')
    plt.xlim([0, 2000])
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.title("Price Distribution")
    st.pyplot(fig)

    st.subheader("Boxplot for Price")
    fig, ax = plt.subplots()
    sns.boxplot(x='price', data=Data_frame, ax=ax)
    plt.title("Price Boxplot")
    st.pyplot(fig)

# BHK Analysis
elif page == "BHK Analysis":
    st.title("BHK Analysis")
    st.subheader("Histogram for BHK")
    fig, ax = plt.subplots()
    ax.hist(Data_frame['BHK'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("BHK")
    plt.ylabel("Count")
    plt.title("BHK Distribution")
    st.pyplot(fig)

    st.subheader("Boxplot for Number of Bedrooms")
    fig, ax = plt.subplots()
    sns.boxplot(x='BHK', data=Data_frame, ax=ax)
    plt.title("Number of Bedrooms Boxplot")
    st.pyplot(fig)

# Scatter Chart
elif page == "Scatter Chart Analysis":
    st.title("Scatter Chart Analysis")
    selected_location = st.sidebar.selectbox("Select Location", Data_frame['location'].unique())
    bhk2 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 2)]
    bhk3 = Data_frame[(Data_frame.location == selected_location) & (Data_frame.BHK == 3)]

    fig, ax = plt.subplots()
    ax.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    ax.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(f"Scatter Chart for {selected_location}")
    plt.legend()
    st.pyplot(fig)

# Bar Plot - Bathroom and BHK
elif page == "Bar Plot Analysis":
    st.title('Bar Plot - Bathroom and BHK')
    fig, ax = plt.subplots()
    sns.barplot(x='BHK', y='bathroom', data=Data_frame, palette='Blues', ax=ax)
    plt.title('Bar Plot - Bathroom and BHK')
    plt.xlabel('BHK')
    plt.ylabel('Bathroom')
    st.pyplot(fig)

# Heatmap
# elif page == "Heatmap Analysis":
#     st.title('Heatmap Analysis')
#     corr_matrix = Data_frame.corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
#     st.pyplot(fig)
elif page == "Heatmap Analysis":
    st.title('Heatmap Analysis')
    
    # Select only numeric columns for correlation matrix
    numeric_columns = Data_frame.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = Data_frame[numeric_columns].corr()
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot(fig)

# Actual vs Predicted Value
elif page == "Actual vs Predicted Value":
    st.title('Actual vs Predicted Value')
    fig, ax = plt.subplots()
    axis = sns.distplot(x=st.session_state['Y_test'], hist=False, color='red', label='Actual Data')
    sns.distplot(x=st.session_state['Y_pred'], hist=False, color='blue', label='Predicted Data', ax=axis)
    plt.xlabel('Price')
    plt.title('Actual Value vs Predicted Value')
    plt.legend(loc='best')
    st.pyplot(fig)

# Regression Plot
elif page == "Regression Plot":
    st.title('Regression Plot - Actual vs Predicted')
    sns.set(color_codes=True)
    sns.set_style("white")
    fig, ax = plt.subplots()
    ax = sns.regplot(x=st.session_state['Y_test'], y=st.session_state['Y_pred'], scatter_kws={'alpha': 0.4})
    ax.set_xlabel('Original Value - Price', fontsize='large', fontweight='bold')
    ax.set_ylabel('Predicted Value - Price', fontsize='large', fontweight='bold')
    ax.set_xlim(35, 135)
    ax.set_ylim(15, 135)
    st.pyplot(fig)
