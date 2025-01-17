import streamlit as st
import pandas as pd
import folium
import folium.plugins
import plotly.express as px
from streamlit_folium import st_folium, folium_static
import branca.colormap as cm
from langchain.chat_models import ChatOpenAI
import geopandas
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#import h3pandas
import time
# New moving to pyspark
from pyspark.sql import SparkSession, functions as F
import h3_pyspark
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from shapely.geometry import Polygon, shape
import re
import json

#from langchain_community.llms import OpenAI
#from langchain.agents.agent_types import AgentType
#from langchain.output_parsers import ResponseSchema, StructuredOutputParser
#from langchain_core.prompts import PromptTemplate
#from langchain.agents import AgentExecutor, Tool
#from langchain.memory import ConversationBufferMemory

#import os
#import math
#import geopy
#from langchain_experimental.agents.agent_toolkits import create_csv_agent
#from langchain.agents.initialize import initialize_agent

def code_dates(data, date_column='sale_date', input_format='yyyy-MM-dd', add_month=True):
    """
    Set date_column to a PySpark date format.
    Default input format: YYYY-MM-DD
    """
    # Convert the date column to PySpark DateType - shouldn't need to as schema already has as date
    #data = data.withColumn(date_column, F.to_date(F.col(date_column), input_format)) shouldn't need
    
    if add_month:
        # Extract year and month, and create a year-month period column
        data = data.withColumn("year_month", F.date_format(F.col(date_column), "yyyyMM").cast("int"))
    
    return data

def prepare_sales_data(
    _data,
    cols_keep=[
        "sale_date",
        "sale_price",
        "beds",
        "sqft",
        "price_per_sqft",
        "bath_full",
        "bath_half",
        "bath_3qtr",
        "year_built",
        "lat",
        "lng",
    ],
):
    # Calculate price_per_sqft
    _data = _data.withColumn("price_per_sqft", F.col("sale_price") / F.col("sqft"))
    # Create year_month column
    # Select specified columns
    _data = _data.select(*cols_keep)
    return _data

@st.cache_resource
def load_data(file,_spark: SparkSession ='', add_h3=False, use_spark = False, lat_col='',lng_col='', format_dates = False, wrangle_sales_data = False):
    if use_spark == False:
        data = pd.read_csv(file)
        if add_h3 == True :
            data = data.h3_pyspark.geo_to_h3(resolution=8, lat_col=lat_col, lng_col=lng_col)
            data = data.h3_pyspark.h3_to_geo_boundary().reset_index()
    else:
        data = _spark.read.csv(file, header=True, inferSchema=True) # type: ignore
        if add_h3 == True:
            data = data.withColumn('h3_8', h3_pyspark.geo_to_h3(lat_col,lng_col,F.lit(8)))
        if wrangle_sales_data == True:
            data = prepare_sales_data(data)
        if format_dates == True:
            # drop existing year_month column if exists
            data = data.drop("year_month")
            # format date column and add year_month
            data = code_dates(data)

    return data

@st.cache_resource
def generate_response(input_text, _dataframes, _container):
    """ "Function to specify AI response and write response to container"""
    dataframe_agent = create_pandas_dataframe_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
        df=_dataframes,
        prefix = 'df1 is listing data containing properties for sale, df2 is place data. If the user says "near" assume they mean within 2 miles. If the response provides items or an item with coordinates, return the coordinates of each returned location as a single pair or list of coordinates in the form [Longitude, Latitude] in square brackets at the end of the response',
        allow_dangerous_code=True,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="openai-tools",
    )

    response = dataframe_agent.invoke(input_text, return_only_outputs=False)['output']
    _container.write(response)
    # Find all coordinates in form [lat, long]
    coordinates =  re.findall('\\[(\\d+, \\d+)\\]', response, re.IGNORECASE)
    if coordinates:
        _container.write(coordinates)
        return coordinates

def coords_to_dict(coord_list):
  '''Function to create a list of dictionaries containing coordinates with keys 'lat' and 'lng'.
  Expects input as list of strings in formart [lat, lng]'''
  dict_new = [] 
  for item in range(0,len(coord_list)):
    dict_new.append({})
    for c in ['lat','lng']:
      for i in enumerate(coord_list[item].split(', ')):
        if c == 'lat' and i[0] == 0:
          print(i[1])
          print(item)
          dict_new[item]['lat'] = float(i[1])
        if c == 'lng' and i[0] == 1:
          dict_new[item]['lng'] = float(i[1])
  return(dict_new)

##@st.cache_data 
## to use widget must be seperated from value. applied to all select containers


@st.cache_data
def get_numerical_summary(_data, variable_name, quartiles = True):
    """Get range of numerical variable in _data"""
    # Calculate min, max, and quantiles using PySpark
    min = _data.agg(F.min(variable_name)).collect()[0][0]
    max = _data.agg(F.max(variable_name)).collect()[0][0]
    array = [min, max]
    if quartiles == True:    
        lq = _data.approxQuantile(variable_name, [0.25], 0.01)[0]
        uq = _data.approxQuantile(variable_name, [0.75], 0.01)[0]
        array.append(lq)
        array.append(uq)
    
    return array

def display_range_filter(summary_array, _container, title, step = 20000):
    """Display continuous integer slider, into specified container"""
    # if length of array is 4, last 2 objects are defaults, else use full range as defaults
    if len(summary_array)==4:
        min_default = summary_array[2]
        max_default = summary_array[3]
    else: 
        min_default = summary_array[0]
        max_default = summary_array[3] 
    # Create slider from range and return values for filters
    selected_min, selected_max = _container.slider(
        title,
        min_value=int(round(summary_array[0], -4)),
        max_value=int(round(summary_array[1], -4)),
        value=(int(round(min_default, -4)),int(round(max_default, -4))),
        step=step
    )
    _container.caption(f'{title} selected: {selected_min} to {selected_max}')
    return selected_min, selected_max

@st.cache_data
def get_distinct_sort(_data, variable_name):
    """Get distinct values based on data values, and sort"""
    # Extract year and month from the date column
    # Edited out as already present in data
    # data = data.withColumn('year_month', F.date_format(F.col(date_column), 'yyyy-MM'))
    # Extract unique year-month values
    distinct_list = _data.select(variable_name).distinct().rdd.flatMap(lambda x: x).collect()
    distinct_list.sort()
    return distinct_list
       
def display_select_slider(list, _container, title, default_min=None):
    """Create slider from list of options,  """
    # Determine min and max values
    min = list[0]
    max = list[-1]

    if default_min:
        min = default_min
    selected_min, selected_max = _container.select_slider(
        title, options=list, value=(min, max)
    )
    _container.caption(f'{title} selected: {selected_min} to {selected_max}')
    return selected_min, selected_max

def filter_sales_data(_data, date_min=None, date_max=None, beds_min=None, beds_max=None):
    """Filter sales data based on filters"""
    
    # Filter by bedroom count if specified
    if beds_min is not None and beds_max is not None:
        _data = _data.filter((F.col("beds") >= int(beds_min)) & (F.col("beds") <= int(beds_max)))
    
    # Filter by date range if specified
    if date_min is not None:
        _data = _data.filter(F.col("year_month") >= int(date_min)) 
    if date_max is not None:
        _data = _data.filter(F.col("year_month") <= int(date_max))
    
    return _data

def filter_listings(
    _data, bounding_box=None, beds_min=None, beds_max=None, price_min=None, price_max=None
):
    """Filter listings based on bedroom filter and/or map bounds"""
    
    # Filter by bounding box if specified
    if bounding_box:
        _data = _data.filter((F.col("latitude") >= bounding_box[0]) & (F.col("latitude") <= bounding_box[2]) &
                           (F.col("longitude") >= bounding_box[1]) & (F.col("longitude") <= bounding_box[3]))
    
    # Filter by bedroom count if specified
    if beds_min is not None and beds_max is not None:
        _data = _data.filter((F.col("bedrooms") >= int(beds_min)) & (F.col("bedrooms") <= int(beds_max)))
    
    # Filter by price range if specified
    if price_min is not None and price_max is not None:
        _data = _data.filter((F.col("price") >= int(price_min)) & (F.col("price") <= int(price_max)))
    
    return _data

def aggregate_sales(_data, _filtered_data):
    """Aggregates sales by month for chart. Uses unfiltered data (all areas) and filtered data (within map bounds)"""
    
    # Aggregate for all sales
    all_sales_monthly = _data.groupBy("year_month") \
                            .agg(
                                F.expr('percentile_approx(price_per_sqft, 0.5)').alias('median_price_per_sqft'),
                                F.count("sqft").alias("observations")
                            ) \
                            .withColumn("Location", F.lit("All areas"))
    
    # Aggregate for filtered sales
    filtered_sales_monthly = _filtered_data.groupBy("year_month") \
                                          .agg(
                                              F.expr('percentile_approx(price_per_sqft, 0.5)').alias('median_price_per_sqft'),
                                              F.count("sqft").alias("observations")
                                          ) \
                                          .withColumn("Location", F.lit("Filtered area"))
    
    # Concatenate the two DataFrames
    sales_monthly = all_sales_monthly.union(filtered_sales_monthly)
    
    return sales_monthly

def aggregate_listings(_data):
    """Aggregates listings by h3 geometry for map colors and small area statistics. Adds geometry and properties column, ready to convert to geojson"""
    # Filter data where pricePerSqft and h3_8 is not null
    data_filtered = _data.filter(F.col("pricePerSqft").isNotNull()) \
                    .filter(F.col("h3_8").isNotNull())
    # Group by H3 index and calculate median pricePerSqft and count
    grouped_data = data_filtered.groupBy("h3_8") \
                                .agg(
                                    F.expr('percentile_approx(pricePerSqft, 0.5)').alias('pricePerSqft'),
                                    F.count("zpid").alias("count")) \
                                .withColumn('geometry',
                                            h3_pyspark.h3_to_geo_boundary(F.col('h3_8'),F.lit(True))) \
                                .withColumn("properties",
                                            F.to_json(F.struct("pricePerSqft","h3_8","count")))

    return grouped_data

def display_sales_aggregate(data, field_name, metric="count", metric_title="Total"):
    """Display summaries of property sales"""
    if metric == "count":
        total = data.count()
    elif metric == "sum":
        total = data.select(F.sum(field_name)).collect()[0][0]
    elif metric == "median":
        total = data.approxQuantile(field_name, [0.5], 0.001)[0]

    # Assuming `st.metric` is used for display purposes
    st.metric(metric_title, "{:,.0f}".format(total))

#@st.cache_data
def display_sales_history(data, monthly_data):
    """ "Display historic sales chart and recent sales"""
    st.subheader("Historic sales prices")
    fig = px.line(
        monthly_data.orderBy(F.col("year_month").asc()),
        x="year_month",
        y="median_price_per_sqft",
        color="Location",
        hover_data = {"median_price_per_sqft":":.2f",
                      "observations":True,
                      "year_month":False,"Location":False},
        color_discrete_map={'All areas': 'lightsteelblue', 'Filtered area': 'steelblue'},
        labels={
            "year_month": "month",
            "median_price_per_sqft": "price per square foot"}
    )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(legend_title_text=None, xaxis_type="category",hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Most recent sales")
    st.table(
    # Sort by 'sale_date' in descending order
        data = data.orderBy(F.col("sale_date").desc()) \
                .withColumnRenamed("sale_date", "date") \
                .withColumnRenamed("sale_price", "price") \
                .withColumnRenamed("year_built", "built") \
                .withColumn('baths', F.col('bath_full') * 1 + F.col('bath_3qtr')+ F.col('bath_half') * 0.5) \
                .drop('bath_full', 'bath_half','bath_3qtr') \
                .toPandas() \
                .set_index('date') \
                .head(n=20) \
                .style.format({"price_per_sqft": "{:,.0f}", "price": "{:,.0f}","baths": "{:,.0f}"})
    )

def display_listings_aggregate(data, field_name, metric="count", metric_title="Total"):
    """Display summary of listings"""
    if metric == "count":
        total = data.count()
    elif metric == "sum":
        total = data.select(F.sum(field_name)).collect()[0][0]
    elif metric == "median":
        total = data.approxQuantile(field_name, [0.5], 0.001)[0]

    st.metric(metric_title, "{:,.0f}".format(total))

def construct_geojson(data, spark_df=True):  
    """Requires dataframe containing fields geometry and properties, as strings representing dict objects"""
    geojson = {
    "type": "FeatureCollection",
    "features": []
    }
    if spark_df == True:
        data = data.collect()
    for row in data:
        feature = {
          "type": "Feature",
          "geometry": { "type" : "Polygon",
                       "coordinates" : [
                           json.loads(row['geometry'])['coordinates']
                           ] },
          "properties": json.loads(row['properties']) 
          }
        geojson["features"].append(feature)
    
    return json.dumps(geojson)

def add_aggregate_layer(data, map):
    geojson = construct_geojson(data)

    #[geojson["features"][0]['geometry']['coordinates'][0][1],geojson["features"][0]['geometry']['coordinates'][0][0]],zoom_start = 14
    # declare colour map for h3 geometry colouring
    linear_sqft = cm.LinearColormap(
        ["green", "yellow", "red"],
        vmin=data.approxQuantile("pricePerSqft", [0.25], 0.001)[0],
        vmax=data.approxQuantile("pricePerSqft", [0.75], 0.001)[0],
    )

    # include aggregate statistics for geometry in tooltip
    tooltip = folium.GeoJsonTooltip(
        fields=["pricePerSqft", "count"],
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

    geo_j = folium.GeoJson(
        data=geojson,
        tooltip=tooltip,
        name="Area average prices",
        style_function=lambda feature: {
            "fillOpacity": 0.2,
            "fillColor": linear_sqft(feature['properties']["pricePerSqft"]),
            "weight": 1,
        },
    )

    geo_j.add_to(map)

def add_places_data(data, map, layer_name='Places'):
    ''' Define layer for points of interest data '''
    places = folium.FeatureGroup(name=layer_name, control=True)
    
    places_data = data.toPandas()
    
    # for each item create pop up and use custom icon
    for i in range(0, len(places_data)):
        # frame which shows address of point
        iframe = folium.IFrame(
            "<style> body {font-family: Tahoma, sans-serif;font-size:10px}</style>"
            + "<b>"
            + places_data.iloc[i]["name"]
            + "</b><br>"
            + places_data.iloc[i]["location.formatted_address"]
        )
        popup = folium.Popup(iframe, min_width=150, max_width=150, max_height=70)
        # Tooltip gives only name
        tooltip = folium.Tooltip(
            "<style> body {font-family: Tahoma, sans-serif;font-weight:bold;font-size:20px;color:black}</style>"
            + places_data.iloc[i]["name"]
        )
        # Use custom icon
        icon = folium.features.CustomIcon(
            "data/noun-baby-cropped.png", icon_size=(30, 30)
        )
        # Set market at coordinates
        folium.Marker(
            location=[
                places_data.iloc[i]["geocodes.main.latitude"],
                places_data.iloc[i]["geocodes.main.longitude"],
            ],
            popup=popup,
            tooltip=tooltip,
            icon=icon,
        ).add_to(places)
    
    places.add_to(map)


def add_listings_data(data, map):
    listings_data = data.toPandas()
    # Colour map for listing markers
    # green to red, using lower and upper quartiles of listing prices
    linear = cm.LinearColormap(
        [(255, 223, 142), (198, 88, 36)],
        vmin=listings_data.price.quantile(0.25),
        vmax=listings_data.price.quantile(0.75),
    )
    # declare cluster for sales
    marker_cluster = folium.plugins.MarkerCluster(
        disableClusteringAtZoom=12, name="Listings"
    ).add_to(map)
    
    # for each item of listings_data create frame with details, image and url
    for i in range(0, len(listings_data)):
        iframe = folium.IFrame(
            "<style> body {font-family: Tahoma, sans-serif;}</style>"
            + "${:,.0f}".format(listings_data.iloc[i]["price"])
            + "<br>"
            + "Beds: "
            + "{:.0f}".format(listings_data.iloc[i]["bedrooms"])
            + " Baths: "
            + "{:.0f}".format(listings_data.iloc[i]["bathrooms"])
            + "<br>"
            + "Living Area: "
            + "{:,.0f}".format(listings_data.iloc[i]["livingArea"])
            + "<br>$/SqFt: "
            + "{:.0f}".format(listings_data.iloc[i]["pricePerSqft"])
            + '<br><img src="'
            + listings_data.iloc[i]["imgSrc"]
            + '" alt="Property image" style="width:200px;height:200px;">'
            + "<br>"
            + "<a "
            + 'href="https://www.zillow.com/homedetails/'
            + str(int(listings_data.iloc[i]["zpid"]))
            + "_zpid"
            + '" target="_blank">See listing</a>',
            width=250,
            height=300,
        )
    
        # present above frames on click
        popup = folium.Popup(
            iframe, min_width=250, max_width=250, min_height=320, max_height=400
        )
    
        # also have tooltips on hover
        tooltip = folium.Tooltip(
            "<style> body {font-family: Tahoma, sans-serif;font-weight:bold;font-size:30px;color:black}</style>"
            + listings_data.iloc[i]["propertyType"].replace("_", " ").title()
            + "<br>"
            + "${:,.0f}".format(listings_data.iloc[i]["price"])
            + "<br>"
            + "Beds: "
            + "{:.0f}".format(listings_data.iloc[i]["bedrooms"])
            + " Baths: "
            + "{:.0f}".format(listings_data.iloc[i]["bathrooms"])
            + "<br>"
            + "Living Area: "
            + "{:,.0f}".format(listings_data.iloc[i]["livingArea"])
            + "<br>$/SqFt: "
            + "{:.0f}".format(listings_data.iloc[i]["pricePerSqft"])
        )
    
        # Place above frame for each individual listing at property coordinates
        folium.CircleMarker(
            location=[
                listings_data.iloc[i]["latitude"],
                listings_data.iloc[i]["longitude"],
            ],
            radius=5,
            popup=popup,
            tooltip=tooltip,
            fill_color=linear(listings_data.iloc[i]["price"]),
            fill_opacity=1,
            color=linear(listings_data.iloc[i]["price"]),
        ).add_to(marker_cluster)
    
    # add cluster layer to map
    map.add_child(marker_cluster)

def create_map(
    _listings_data, _aggregate_listing_data, _places_data, coordinates, places_name="places"
):
    """Function to set up map, add layers, and run in streamlit"""
    if coordinates:
        st.caption(coordinates)
        listings_map = folium.Map(
        [coordinates[0]['lat'],coordinates[0]['lng']], zoom_start=14   
    ) 
    else:
    # Set map centre as mean of coordinates, set default zoom
        listings_map = folium.Map(
            [_listings_data.agg(F.mean('latitude')).collect()[0][0], _listings_data.agg(F.mean('longitude')).collect()[0][0]], zoom_start=14   
    )

    # Add each Layer
    add_aggregate_layer(data = _aggregate_listing_data, map = listings_map)
    add_places_data(data = _places_data, map = listings_map, layer_name = places_name)
    add_listings_data(data = _listings_data, map = listings_map)
        
    ## add layer control option
    folium.LayerControl().add_to(listings_map)
    st.session_state.map = listings_map
# return map
    return st.session_state.map


    # Create map with st_folium

    
def get_bounding_box(map):
        # Create bounding box from map bounds
    bounding_box = [
                map["bounds"]["_southWest"]["lat"],
                map["bounds"]["_southWest"]["lng"],
                map["bounds"]["_northEast"]["lat"],
                map["bounds"]["_northEast"]["lng"],
        ]
    return bounding_box

