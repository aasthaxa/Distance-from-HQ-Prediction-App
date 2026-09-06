
import streamlit as st
import pandas as pd
import joblib

# Load the saved linear regression model
loaded_lr_model = joblib.load('linear_regress_model.sav')

st.title('Distance from HQ Prediction App')

st.write("Enter the feature values to predict the 'Distance from HQ'.")

# Display model coefficients and intercept
st.subheader('Model Insights')
st.write(f"**Intercept:** {loaded_lr_model.intercept_:.2f}")
st.write("**Coefficients:**")

# Feature names from WorkWell_stepwise_data.csv, excluding 'Dist_from_HQ'
feature_names = [
    'Years_Experience', 'Cust_Satisfaction', 'Client_Meetings', 'Marketing_Spend',
    'Territory_Size', 'Portfolio_Size', 'Training_Sessions', 'Digital_Adoption',
    'Peer_Collab', 'Monthly_Sales'
]

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': loaded_lr_model.coef_})
st.table(coef_df)

st.subheader('Input Feature Values')

# Create input fields for features with reasonable default values based on df.describe()
years_experience = st.number_input('Years_Experience', min_value=0.0, max_value=20.0, value=9.0, step=0.1)
cust_satisfaction = st.number_input('Cust_Satisfaction', min_value=0.0, max_value=10.0, value=5.0, step=0.01)
client_meetings = st.number_input('Client_Meetings', min_value=0.0, max_value=30.0, value=15.0, step=0.1)
marketing_spend = st.number_input('Marketing_Spend', min_value=0.0, max_value=200.0, value=100.0, step=0.1)
territory_size = st.number_input('Territory_Size', min_value=0.0, max_value=1000.0, value=500.0, step=1.0)
portfolio_size = st.number_input('Portfolio_Size', min_value=0.0, max_value=25.0, value=12.0, step=0.1)
training_sessions = st.number_input('Training_Sessions', min_value=0, max_value=10, value=5, step=1)
digital_adoption = st.number_input('Digital_Adoption', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
peer_collab = st.number_input('Peer_Collab', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
monthly_sales = st.number_input('Monthly_Sales', min_value=0.0, max_value=200.0, value=120.0, step=0.1)

# Create a DataFrame for the input
input_data = pd.DataFrame([[
    years_experience, cust_satisfaction, client_meetings, marketing_spend,
    territory_size, portfolio_size, training_sessions, digital_adoption,
    peer_collab, monthly_sales
]], columns=feature_names)

# Make prediction
if st.button('Predict Distance'):
    prediction = loaded_lr_model.predict(input_data)[0]
    st.success(f'Predicted Distance from HQ: {prediction:.2f} units')
