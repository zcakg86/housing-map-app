
from functions import *
import streamlit as st
import pandas as pd
import folium
import folium.plugins
import plotly.express as px
from streamlit_folium import st_folium
import branca.colormap as cm
import h3pandas
from langchain.llms import OpenAI
import os

from langchain_experimental.agents.agent_toolkits import create_csv_agent
APP_TITLE = 'Housing search'
APP_SUB_TITLE = 'Produced by Marie Thompson'

def main():
    st.set_page_config(APP_TITLE, layout = 'wide')
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    # load data
    sales_data = pd.read_csv('data/sales_2021_on_geo.csv')
    sales_data = sales_data.h3.geo_to_h3(resolution = 8, lat_col = 'lat', lng_col = 'lng')
    sales_data = sales_data.h3.h3_to_geo_boundary()

    listings_data = pd.read_csv('data/all-listings.csv')
    listings_data = listings_data.h3.geo_to_h3(resolution = 8, lat_col = 'latitude', lng_col = 'longitude')
    listings_data = listings_data.h3.h3_to_geo_boundary()

    daycare_data = pd.read_csv('data/daycares.csv')
    # display filters and map
    beds_min, beds_max = display_bedroom_filter(sales_data)
    date_min, date_max = display_time_filters(sales_data,'sale_date')

    # Display metrics
    col_left,col_right = st.columns(2)
    with col_left:
        location_chat = st.chat_input(placeholder="Ask about a location")
        if location_chat:
            st.write(f"User has sent the following prompt: {location_chat}")
            generate_response(location_chat,
                              dataframes = [daycare_data.loc[:,['name','geocodes.main.latitude','geocodes.main.longitude']],
                                            listings.loc[:,['price','latitude','longitude','propertyType','bedrooms']]])

        st.header('Listings map')
        grouped_listings = aggregate_listings(listings_data)
        bounding_box = display_map(listings_data=listings_data, aggregate_listing_data=grouped_listings, places_data = daycare_data, places_name = 'Day cares')

        # listings
        listings = filter_listings(data=listings_data, bounding_box=bounding_box, beds_min=beds_min, beds_max=beds_max)
        col1, col2 = st.columns(2)
        with col1:
            display_listings_aggregate(data=listings, field_name='price', metric='count', metric_title=f'Total listings')
        with col2:
            display_listings_aggregate(data=listings, field_name='price', metric='median', metric_title=f'Median listing price')

    with col_right:
        st.header("Sales history")
        # filter data with bounding_box and dates/beds and calculate listings
        # sales
        sales = filter_sales_data(sales_data, bounding_box = '', date_min=int(date_min.replace('-', '')),
                                  date_max=int(date_max.replace('-', '')), beds_min=beds_min, beds_max=beds_max)
        if bounding_box:
            filtered_sales = sales[sales['lat'].between(bounding_box[0],bounding_box[2])]
            filtered_sales = sales[sales['lng'].between(bounding_box[1], bounding_box[3])]
        monthly_sales = aggregate_sales(data = sales, filtered_data = filtered_sales)
        col3,col4 = st.columns(2)
        with col3:
            display_sales_aggregate(data=sales,
                                    field_name='sale_price', metric='count',
                                    metric_title=f'Total sales')
        with col4:
            display_sales_aggregate(data=sales, field_name='sale_price', metric='median',
                                    metric_title=f'Median sale price')

        display_sales_history(data=sales, monthly_data = monthly_sales)

if __name__ == "__main__":
    main()
