#writefile delivery_app.py
import streamlit as st
import catboost
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model_delivery.pkl", "rb"))


st.title("🚀 ETA Prediction of Delivery")
st.write("Enter details to predict the Estimated Time of Arrival (ETA).")

# Input fields
region_id = st.number_input("Region ID", value=0)
courier_id = st.number_input("Courier ID", value=0)
lng = st.number_input("Longitude", value=0.0)
lat = st.number_input("Latitude", value=0.0)
aoi_id = st.number_input("AOI ID", value=0)
aoi_type = st.number_input("AOI Type", value=0)
accept_gps_lng = st.number_input("Accept GPS Longitude", value=0.0)
accept_gps_lat = st.number_input("Accept GPS Latitude", value=0.0)
delivery_gps_lng = st.number_input("Delivery GPS Longitude", value=0.0)
delivery_gps_lat = st.number_input("Delivery GPS Latitude", value=0.0)
distance = st.number_input("Distance", value=0.0)
accept_hour_of_day = st.number_input("Accept Hour of Day", value=0)
accept_day_of_week = st.number_input("Accept Day of Week", value=0)
delivery_hour_of_day = st.number_input("Delivery Hour of Day", value=0)
delivery_day_of_week = st.number_input("Delivery Day of Week", value=0)
delivery_distance = st.number_input("Delivery Distance", value=0.0)
log_delivery_distance = st.number_input("Log Delivery Distance", value=0.0)
pca1 = st.number_input("PCA1", value=0.0)
pca2 = st.number_input("PCA2", value=0.0)

# Prediction
if st.button("Predict ETA"):
    input_features = np.array([[region_id, courier_id, lng, lat, aoi_id, aoi_type,
                                accept_gps_lng, accept_gps_lat, delivery_gps_lng, delivery_gps_lat,
                                distance, accept_hour_of_day, accept_day_of_week,
                                delivery_hour_of_day, delivery_day_of_week, delivery_distance,
                                log_delivery_distance, pca1, pca2]])
    
    prediction = model.predict(input_features)
    st.success(f"Predicted ETA: {prediction[0]:.2f} minutes")
