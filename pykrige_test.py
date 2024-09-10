from pykrige.ok import OrdinaryKriging
import numpy as np
import plotly_express as px
import pandas as pd
import streamlit as st

def krige(data):
    # Extract coordinates and price values
    lats = data['lat'].values
    lons = data['lng'].values
    prices = data['ppsf_adj'].values

    # Create the Ordinary Kriging model
    OK = OrdinaryKriging(
        x=lons, y=lats, z=prices,
        variogram_model='gaussian',  # Can be 'linear', 'power', 'gaussian', 'spherical', etc.
        verbose=True,
        enable_plotting=False
    )

    # Define a grid where you want to predict prices
    grid_lon = np.linspace(min(lons), max(lons), 40)  # Adjust the range and resolution as needed
    grid_lat = np.linspace(min(lats), max(lats), 40)

    # Perform Kriging on the grid, predicted values (z) and variance (ss)
    z, ss = OK.execute('grid', grid_lon, grid_lat)

    X, Y = np.meshgrid(grid_lon, grid_lat)

    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    grid_lon = grid_lon.flatten()
    grid_lat = grid_lat.flatten()
    z = z.flatten()
    plot_data = pd.DataFrame({
        'Longitude': grid_lon,
        'Latitude': grid_lat,
        'Predicted Price': z
    })
    # fig = px.scatter_3d(
    #     plot_data, 
    #     x='Longitude', 
    #     y='Latitude', 
    #     z='Predicted Price', 
    #     color='Predicted Price', 
    #     color_continuous_scale='Viridis',
    #     title='Kriging Predicted Prices',
    #     labels={'Longitude': 'Longitude', 'Latitude': 'Latitude', 'Predicted Price': 'Predicted Price'}
    # )

    # st.plotly_chart(fig)

    return plot_data