import streamlit as st

def welcome_page():
    st.title("House Price Prediction App")
    
    # Frontend design
    st.image("logo.png", caption="Welcome To House Price Predictor App", use_column_width=True)

    st.write(
        "Welcome to the Bengaluru House Price Prediction App! This app is designed to provide estimates of house prices in Bengaluru based on various features. "
        "Explore the features and get an idea of the potential price of your dream home."
    )

    # About the project
    st.header("About the Project")
    st.write(
        "This project utilizes machine learning techniques to predict house prices in Bengaluru. The model has been trained on a dataset containing various features "
        "such as total square feet, number of balconies, number of bathrooms, BHK (Bedrooms, Hall, Kitchen), area type, and location."
    )

    st.write(
        "The dataset has been preprocessed to handle outliers and ensure accurate predictions. The model is a Linear Regression model trained on a cleaned dataset. "
        "It takes user inputs and provides an estimate of the house price based on the learned patterns from the training data."
    )

    st.write(
        "Feel free to use the app to get an approximate value of a house in Bengaluru. Please note that the predictions are based on historical data and patterns, "
        "and actual prices may vary."
    )

    st.success("Let's get started! Use the navigation bar on the left to input details and predict house prices.")

if __name__ == "__main__":
    welcome_page()
