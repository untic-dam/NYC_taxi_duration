from random import random
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import random

#functions
def random_lat():
    return random.uniform(lat_magnitude[0], lat_magnitude[1])

def random_lon():
    return random.uniform(lon_magnitude[0], lon_magnitude[1])

def random_location():
    random_pickup = {'lat': random_lat(), 'lon': -random_lon()}
    random_dropoff = {'lat': random_lat(), 'lon': -random_lon()}

    df_pick = pd.DataFrame(random_pickup, index=[0])
    df_drop = pd.DataFrame(random_dropoff, index=[0])

    frames = [df_pick, df_drop]
    df_location = pd.concat(frames)

    return df_location

def convertToMin(duration):
    min = int(duration//60)
    sec = int(duration%60)

    return f'{min} min {sec}'

def computeDist(pickup, dropoff):
    
    dist = geodesic(pickup, dropoff).meters
    dist = round(dist)
    
    return dist

#init
data = pd.read_csv('data.csv', nrows=1000)
data = data[['trip_duration', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour', 'weekday']]

lat_magnitude = [40.70, 40.8]
lon_magnitude = [73.95, 74.06]

lat_ = (40.80, 40.9)
lon_ = (73.95, 74.06)

df_location = random_location()
pick_lat = df_location.iloc[0,0]
pick_lon = df_location.iloc[0,1]
drop_lat = df_location.iloc[1,0]
drop_lon = df_location.iloc[1,1]

#Title
st.write('# NYC taxi duration estiamtor')

#Maps
st.write('## Map: ')
st.map(df_location)

#Button
button_random = st.button('random location')
if button_random:
    df_location = random_location()
    pick_lat = df_location.iloc[0,0]
    pick_lon = df_location.iloc[0,1]
    drop_lat = df_location.iloc[1,0]
    drop_lon = df_location.iloc[1,1]
else:
    pass

#model
model = joblib.load('model_nyc_taxi.joblib')

pickup = (pick_lat, pick_lon)
dropoff = (drop_lat, drop_lon)

trip = {'distance': computeDist(pickup, dropoff),
        'weekday': 'Friday',
        'hour': 20,
        'pickup_longitude': pick_lon,
        'pickup_latitude': pick_lat,
        'dropoff_longitude': drop_lon,
        'dropoff_latitude': drop_lat}

trip_df = pd.DataFrame(trip, index=[0])

duration = model.predict(trip_df)

st.write("# Trip Duration: ")
st.write(convertToMin(duration))

st.write("# Training Data")
st.write(data)