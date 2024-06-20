import streamlit as st
import pandas as pd
import folium
import folium.plugins
import plotly.express as px
from streamlit_folium import st_folium
import branca.colormap as cm
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import geopandas
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.agents import (AgentExecutor, Tool)
from langchain.memory import ConversationBufferMemory
from functions import *
import h3pandas
import os
import math
import geopy
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.initialize import initialize_agent

#response_schemas = [
#    ResponseSchema(name="answer", description="answer to the user's question"),
#    ResponseSchema(
#        name="lat",
#        description="latitude of object in answer"
#    ),
#    ResponseSchema(
#        name="long",
#        description="longitude of object in answer"
#    ),
#    ResponseSchema(
#        name="name",
#        description="name of object in answer"
#    ),
#]
#output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#format_instructions = output_parser.get_format_instructions()


def generate_response(input_text, dataframes, container):

    dataframe_agent = create_pandas_dataframe_agent(
        llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0125'),
        df=dataframes,
        allow_dangerous_code = True,
        verbose = True,
        handle_parsing_errors = True,
        agent_type="openai-tools")
    #tools = [
    #    Tool(
    #        name="dataframe_agent",
    #        func=dataframe_agent.invoke,
    #        description="Respond based on the dataframe_agent only."
    #    ),
    #]
    #agent = initialize_agent(
    #    llm=OpenAI(temperature=0, model="davinci-002"),
    #    tools = tools,
    #    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #    verbose=True
    #)

    container.write(dataframe_agent.invoke(input_text, return_only_outputs=False))

def display_bedroom_filter(data):
    beds_list = list(data['beds'].unique())
    beds_list.sort()
    beds_min = beds_list[0]
    beds_max = beds_list[len(beds_list)-1]
    beds_min, beds_max = st.sidebar.select_slider('Bedrooms', options=beds_list, value = (beds_min,beds_max))
    #st.header(f'Showing only {beds_min} to {beds_max} bedroom sales and listings')
    return beds_min, beds_max

def display_time_filters(data, date_column):
    year_month_list = list(data[date_column].str[0:7].unique())
    year_month_list.sort()
    min = year_month_list[0]
    max = year_month_list[len(year_month_list)-1]
    min, max = st.sidebar.select_slider('Monthly filter for sales history', options=year_month_list, value=('2023-01', max))
    #st.header(f'Sale date filter: {min} to {max}')
    return min, max

def filter_sales_data(data, date_min, date_max, beds_min, beds_max):
    # removed bounding_box argument as not used in current form
    #if bounding_box:
    #    data = data[data['lat'].between(bounding_box[0],bounding_box[2])]
    #    data = data[data['lng'].between(bounding_box[1], bounding_box[3])]
    if beds_min & beds_max:
        data = data[data['beds'].between(int(beds_min),int(beds_max))]
    if date_min & date_max:
        data['year_month']=data['sale_date'].str[0:7].str.replace('-', '').astype('int')
        data = data[data['year_month'].between(date_min, date_max)]
    data.reset_index(inplace=True)
    data['price_per_sqft']=data['sale_price']/data['sqft']
    data = data.loc[:,['year_month','sale_date','sale_price', 'beds', 'sqft', 'price_per_sqft','bath_full', 'bath_half', 'bath_3qtr', 'year_built','lat','lng']]

    return data

def filter_listings(data, bounding_box, beds_min, beds_max):
    if bounding_box:
        data = data[data['latitude'].between(bounding_box[0],bounding_box[2])]
        data = data[data['longitude'].between(bounding_box[1], bounding_box[3])]
    if beds_min & beds_max:
        data = data[data['bedrooms'].between(int(beds_min),int(beds_max))]
    return data

def aggregate_sales(data, filtered_data):
    all_sales_monthly = data.groupby('year_month').agg({'price_per_sqft': 'median', 'sqft': 'size'})\
                        .rename(columns={'sqft': 'observations'}).reset_index()
    all_sales_monthly['Location']='All areas'
    # calculate monthly sales for filtered data
    filtered_sales_monthly = filtered_data.groupby('year_month').agg({'price_per_sqft': 'median', 'sqft': 'size'})\
                             .rename(columns={'sqft': 'observations'}).reset_index()
    filtered_sales_monthly['Location']='Map area'
    sales_monthly = pd.concat([filtered_sales_monthly, all_sales_monthly]).reset_index()
    return sales_monthly

def aggregate_listings(data):
    ## Now create aggregates sales prices by H3index
    grouped_data = geopandas.GeoDataFrame(data.loc[pd.notnull(data.pricePerSqft)]\
                                                .groupby(['h3_08', 'geometry'])\
                                                .agg({'pricePerSqft': 'median', 'zpid': 'size'})\
                                                .rename(columns={'zpid': 'count'}).reset_index())
    return grouped_data

def display_sales_aggregate(data, field_name, metric = 'count', metric_title = 'Total'):
    if metric == 'count':
        total = data[field_name].count()
    if metric == 'sum':
        total = data[field_name].sum()
    if metric == 'median':
        total = data[field_name].quantile(0.5)
    st.metric(metric_title,"{:,.0f}".format(total))

def display_sales_history(data, monthly_data):
    # get all sales aggregates

    st.subheader("Median sale price per sqft")
    fig = px.line(monthly_data, x="year_month", y="price_per_sqft", color = 'Location',
                  labels = {'year_month':'month','price_per_sqft':'price per square foot','observations':'number of sales'})
    fig.update_layout(legend_title_text=None, xaxis_type='category')
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Most recent sales")

    st.table(data.sort_values(by=['sale_date'], ascending = False).set_index('sale_date')\
             .head(20).style.format({"price_per_sqft": "{:,.0f}","sale_price": "{:,.0f}"}))
    
def display_listings_aggregate(data, field_name, metric = 'count', metric_title = 'Total'):
    if metric == 'count':
        total = data[field_name].count()
    if metric == 'sum':
        total = data[field_name].sum()
    if metric == 'median':
        total = data[field_name].quantile(0.5)
    st.metric(metric_title,"{:,.0f}".format(total))
    
def display_map(listings_data, aggregate_listing_data, places_data, places_name = 'places'):

    # green to red, using lower and upper quartiles
    linear = cm.LinearColormap(["green", "yellow", "red"], vmin=listings_data.price.quantile(0.25),\
                               vmax=listings_data.price.quantile(0.75))
    listings_map = folium.Map([listings_data.latitude.mean(), listings_data.longitude.mean()],zoom_start = 12)

    marker_cluster = folium.plugins.MarkerCluster(disableClusteringAtZoom=14, name='Listings').add_to(listings_map)

    for i in range(0, len(listings_data)):
        iframe = folium.IFrame('<style> body {font-family: Tahoma, sans-serif;}</style>' + \
                               '${:,.0f}'.format(listings_data.iloc[i]['price']) + '<br>' + 'Beds: ' + \
                               "{:.0f}".format(listings_data.iloc[i]['bedrooms']) + ' Baths: ' + \
                               "{:.0f}".format(listings_data.iloc[i]['bathrooms']) + '<br>' + 'Living Area: ' + \
                               "{:,.0f}".format(listings_data.iloc[i]['livingArea']) + '<br>$/SqFt: ' + \
                               "{:.0f}".format(listings_data.iloc[i]['pricePerSqft']) + '<br><img src="' + \
                               listings_data.iloc[i]['imgSrc'] + \
                                '" alt="Property image" style="width:200px;height:200px;">' + \
                                '<br>' + '<a ' + 'href="https://www.zillow.com/homedetails/' + str(
                                int(listings_data.iloc[i]['zpid'])) + '_zpid' + '" target="_blank">See listing</a>', \
                               width=250, height=300)

        popup = folium.Popup(iframe, min_width=250, max_width=250, min_height=320, max_height=400)
        tooltip = folium.Tooltip('<style> body {font-family: Tahoma, sans-serif;font-weight:bold;font-size:30px;color:black}</style>'+\
                                 listings_data.iloc[i]['propertyType'].replace('_',' ').title()+'<br>'+\
                                 '${:,.0f}'.format(listings_data.iloc[i]['price']) + '<br>' + 'Beds: ' + \
                                 "{:.0f}".format(listings_data.iloc[i]['bedrooms']) + ' Baths: ' + \
                                 "{:.0f}".format(listings_data.iloc[i]['bathrooms']) + '<br>' + 'Living Area: ' + \
                                 "{:,.0f}".format(listings_data.iloc[i]['livingArea']) + '<br>$/SqFt: ' + \
                                 "{:.0f}".format(listings_data.iloc[i]['pricePerSqft']))
        folium.CircleMarker(
            location=[listings_data.iloc[i]['latitude'], listings_data.iloc[i]['longitude']],
            radius=5,
            popup=popup,
            tooltip = tooltip,
            fill_color=linear(listings_data.iloc[i]['price']),
            fill_opacity=1,
            color=linear(listings_data.iloc[i]['price'])).add_to(marker_cluster)

    listings_map.add_child(marker_cluster)

    linear_sqft = cm.LinearColormap(["green", "yellow", "red"], vmin=aggregate_listing_data.pricePerSqft.quantile(0.25),
                                    vmax=aggregate_listing_data.pricePerSqft.quantile(0.75))

    tooltip = folium.GeoJsonTooltip(
        fields=["pricePerSqft", 'count'],
        aliases=["Price per Sq Ft (USD): ", "Number of Listings: "],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=200,
    )

    geo_j = folium.GeoJson(data=aggregate_listing_data,
                           tooltip=tooltip,
                           name='Area average prices',
                           style_function=lambda feature: {
                               "fillOpacity": 0.2,
                               "fillColor": linear_sqft(feature['properties']['pricePerSqft']),
                               "weight": 0,
                           })

    geo_j.add_to(listings_map)

    #for feature in geo_j.data.values.__self__['features']:
    #    h3_08_name = feature['properties']['h3_08']

    # Add places
    places = folium.FeatureGroup(name=places_name, control=True).add_to(listings_map)
    
    for i in range(0, len(places_data)):
        iframe = folium.IFrame('<style> body {font-family: Tahoma, sans-serif;}</style>' + \
                               '<b>' + places_data.iloc[i]['name'] + '</b><br>' + places_data.iloc[i][
                                   'location.formatted_address'],
                               width=200, height=100)
        popup = folium.Popup(iframe, min_width=200, max_width=220, max_height=200)
        tooltip = folium.Tooltip('<style> body {font-family: Tahoma, sans-serif;font-weight:bold;font-size:20px;color:black}</style>'+places_data.iloc[i]['name'])
        icon = folium.features.CustomIcon('data/noun-baby-6828055.png', icon_size=(40, 40))
        folium.Marker(
            location=[places_data.iloc[i]['geocodes.main.latitude'], places_data.iloc[i]['geocodes.main.longitude']],
            popup=popup,
            tooltip=tooltip,
            icon=icon
        ).add_to(places)

    places.add_to(listings_map)

    folium.LayerControl().add_to(listings_map)

    st_map = st_folium(listings_map, use_container_width=True, height = 600)

    h3_08_name = ''
    bounding_box = ''
    
    ## not used: not using h3_8 click
    ## if st_map['last_active_drawing']:
    ##    h3_08_name = st_map['last_active_drawing']['properties']['h3_08']

    if st_map['bounds']: # map should have bounds, to use in filter
        bounding_box = [st_map['bounds']['_southWest']['lat'],
                        st_map['bounds']['_southWest']['lng'],
                        st_map['bounds']['_northEast']['lat'],
                        st_map['bounds']['_northEast']['lng']]
    return bounding_box ##, h3_08_name

