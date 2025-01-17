from functions.webmap import *
### means to do
APP_TITLE = "Housing search map app"
APP_SUB_TITLE = "Produced by Marie Thompson using streamlit. Data from Zillow, Foursquare API and King County Department of Assessments"

spark = SparkSession.builder \
    .config('spark.driver.host',"localhost") \
    .appName("housing_data") \
    .getOrCreate()

def main():
    # Start declaring streamlit app content
    # Set page name
    st.set_page_config(APP_TITLE, layout="wide")
    # Remove white space at top of page
    st.markdown(
        """<style> .block-container {padding-top: 1rem;}</style>""",
        unsafe_allow_html=True,
    )

    # Place title and subtitle into title container
    header, progress = st.container(border=False).columns((5,1))
    with header:
        st.title(APP_TITLE)
        st.caption(APP_SUB_TITLE)
    with progress: 
        empty = st.header("")
        updates = st.empty()
    # record set up time
    start_time = time.time()

    # load data

    sales_data = load_data(file = "data/sales_2021_on_geo.csv", _spark = spark, 
                           use_spark = True, add_h3 = True, lat_col="lat", lng_col="lng",
                           format_dates = True, wrangle_sales_data = True)
    listings_data = load_data(file = "data/all-listings.csv", _spark = spark, use_spark = True, add_h3 = True, lat_col="latitude", lng_col="longitude")
    grouped_listings = aggregate_listings(listings_data)

    daycare_data = load_data("data/daycares.csv", _spark = spark, use_spark = True)

    data_loaded_time = time.time()
    updates.status(label="%s seconds to load data" % round(data_loaded_time - start_time,2))
    # Display filters in sidebar
    with st.sidebar:
        st.header("Apply filters")
    beds_list = get_distinct_sort(sales_data,"beds")
    beds_min, beds_max = display_select_slider(beds_list, st.sidebar, title = "Bedrooms")

    date_list = get_distinct_sort(sales_data, "year_month")
    date_min, date_max = display_select_slider(date_list, st.sidebar, 'Sale date', default_min= 202201)

    price_array = get_numerical_summary(listings_data, "price")
    price_min, price_max = display_range_filter(price_array, st.sidebar, title = "Price range")

    # Display chat in sidebar
    with st.sidebar:
        st.header("AI assist")
    chat = st.sidebar.container()
    location_chat = chat.chat_input(placeholder="Chat about listings")
    coordinates = None
    if location_chat:
        chat.write(f"User has sent the following prompt: {location_chat}")
        coordinates = generate_response(
            location_chat,
            _dataframes=[
                daycare_data.withColumnRenamed("geocodes.main.latitude","latitude") \
                .withColumnRenamed("geocodes.main.longitude", "longitude")
                .select('name','latitude','longitude')
                .toPandas(),
                listings_data.select("price", "latitude", "longitude", "propertyType", "bedrooms")
                .toPandas()
                ],
            #  dataframes = [daycare_data,listings_data.reset_index()],
            _container=chat,
            )

    # Filter listings based on bedroom and price slider
    listings_data = filter_listings(
        _data=listings_data,
        beds_min=beds_min,
        beds_max=beds_max,
        price_min=price_min,
        price_max=price_max,
    )

    # Filter sales data also on date filter
    sales_data = filter_sales_data(
        sales_data,
        date_min=date_min,
        date_max=date_max,
        beds_min=beds_min,
        beds_max=beds_max,
    )

    slicer_filter_time = time.time()
    updates.status("%s seconds to filter data on slicers" % round(slicer_filter_time - data_loaded_time,2))


    # Break page into two columns (left side containing map is wider)
    col_left, col_right = st.columns((2, 1))
    with col_left:
        # Create container with aggregate data to sit above map
        top_container = st.container(border=False)
        col1, col2, col3, col4 = top_container.columns(4)
        # Create container with caption and button to set filters to map bounds
        next_container = st.container(border=False)
        caption, map_filter_button = next_container.columns((2,1)) 
        with caption: 
            st.caption(
            "Metrics above and historic sales prices are based the bedroom and date filters selected, and the area shown in the map when the reset button is pressed"
        )
        with map_filter_button: 
            st.button('Set filter to current map bounds')

        # display map and return map bounds to filter data
        #st.write(grouped_listings)
        complete_map = create_map(
            _listings_data=listings_data,
            _aggregate_listing_data=grouped_listings,
            _places_data=daycare_data,
            coordinates = coordinates,
            places_name="Day cares"
        )
        # launch streamlit object of map
        st_map = folium_static(complete_map, width=600, height=600)
        map_loaded_time = time.time()
        updates.status("%s seconds to load map" % round(map_loaded_time - slicer_filter_time,2))
        
        bounding_box = ""
        if map_filter_button:
            bounding_box = get_bounding_box(map=st_map)
            updates.status(bounding_box)
        # filter listings based on map bounds for listing metrics
            # filter for 'This area' in sale price chart, and recent sales
            listings = filter_listings(_data=listings_data, bounding_box=bounding_box)
            # filter sales based on map bounds and bedroom/date filter
            # filter data with bounding_box and dates/beds
            listings_filter_time = time.time()
            updates.status("%s seconds to filter listings on bounding box" % round(listings_filter_time - map_loaded_time,2))

            area_filtered_sales = sales_data.filter(
                (sales_data['lat'].between(bounding_box[0], bounding_box[2])) &
                (sales_data['lng'].between(bounding_box[1], bounding_box[3]))
        )
        else: 
            area_filtered_sales = sales_data
        
        sales_filter_time = time.time()
        updates.status("%s seconds to filter sales on bounding box" % round(sales_filter_time - listings_filter_time,2))

        #sales_data.columns
        #['sale_date', 'sale_price', 'beds', 'sqft', 'price_per_sqft', 'bath_full', 'bath_half', 'bath_3qtr', 'year_built', 'lat', 'lng', 'year_month']
        # Place listing metrics
        with col1:
            display_listings_aggregate(
                data=listings,
                field_name="price",
                metric="count",
                metric_title=f"Total listings",
            )
        with col2:
            display_listings_aggregate(
                data=listings,
                field_name="price",
                metric="median",
                metric_title=f"Median listing price",
            )
        with col3:
            display_sales_aggregate(
                data=area_filtered_sales,
                field_name="sale_price",
                metric="count",
                metric_title=f"Total sales",
            )
        with col4:
            display_sales_aggregate(
                data=area_filtered_sales,
                field_name="sale_price",
                metric="median",
                metric_title=f"Median sale price",
            )



    with col_right:  # presenting sales history data
        # aggregate by month for displaying chart
        monthly_sales = aggregate_sales(_data=sales_data, _filtered_data=area_filtered_sales)
        # display chart and table of recent sales
        display_sales_history(data=area_filtered_sales, monthly_data=monthly_sales)

    end_time = time.time()
    updates.status("%s seconds to load everything" % round(end_time - start_time,2))

if __name__ == "__main__":
    main()
