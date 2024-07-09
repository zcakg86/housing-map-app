from functions import *

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

    # record set up time
    start_time = time.time()

    # load data
    sales_data = load_data(file = "data/sales_2021_on_geo.csv", _spark = spark, use_spark = True, add_h3 = True, lat_col="lat", lng_col="lng")

    listings_data = load_data(file = "data/all-listings.csv", _spark = spark, use_spark = True, add_h3 = True, lat_col="latitude", lng_col="longitude")
    grouped_listings = aggregate_listings(listings_data)

    daycare_data = load_data("data/daycares.csv", _spark = spark, use_spark = True)



    # Remove white space at top of page
    st.markdown(
        """<style> .block-container {padding-top: 1rem;}</style>""",
        unsafe_allow_html=True,
    )

    # Place title and subtitle into title container
    titles = st.container(border=False)
    titles.title(APP_TITLE)
    titles.caption(APP_SUB_TITLE)

    # Display filters in sidebar
    with st.sidebar:
        st.header("Apply filters")
    beds_min, beds_max = display_bedroom_filter(sales_data, st.sidebar)
    date_min, date_max = display_time_filter(
        sales_data, "sale_date", st.sidebar, "2022-01"
    )
    price_min, price_max = display_price_filter(listings_data, "price", st.sidebar)

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
    if coordinates:
        chat.write(coordinates)
        chat.write(coords_to_dict(coordinates))
    # Filter listings based on bedroom and price slider
    listings_data = filter_listings(
        data=listings_data,
        beds_min=beds_min,
        beds_max=beds_max,
        price_min=price_min,
        price_max=price_max,
    )
    # Filter sales data also on date filter
    sales_data = filter_sales_data(
        sales_data,
        date_min=int(date_min.replace("-", "")),
        date_max=int(date_max.replace("-", "")),
        beds_min=beds_min,
        beds_max=beds_max,
    )

    # Break page into two columns (left side containing map is wider)
    col_left, col_right = st.columns((2, 1))
    with col_left:
        # Create container with aggregate data to sit above map
        container = st.container(border=False)
        # display map and return map bounds to filter data
        #st.write(grouped_listings)
        bounding_box = display_map(
            listings_data=listings_data,
            aggregate_listing_data=grouped_listings,
            places_data=daycare_data,
            coordinates = coordinates,
            places_name="Day cares"
        )
        
        # filter listings based on map bounds for listing metrics
        listings = filter_listings(data=listings_data, bounding_box=bounding_box)
        # filter sales based on map bounds and bedroom/date filter
        # filter data with bounding_box and dates/beds
        sales_data = prepare_sales_data(sales_data)
        # filter for 'This area' in sale price chart, and recent sales
        filtered_sales = sales_data.filter(
            (sales_data['lat'].between(bounding_box[0], bounding_box[2])) &
            (sales_data['lng'].between(bounding_box[1], bounding_box[3]))
        )

        col1, col2, col3, col4 = container.columns(4)
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
                data=filtered_sales,
                field_name="sale_price",
                metric="count",
                metric_title=f"Total sales",
            )
        with col4:
            display_sales_aggregate(
                data=filtered_sales,
                field_name="sale_price",
                metric="median",
                metric_title=f"Median sale price",
            )
        container.caption(
            "Metrics above and historic sales prices are based on map area displayed in window, and the bedroom and date filters selected."
        )

    with col_right:  # presenting sales history data
        # aggregate by month for displaying chart
        monthly_sales = aggregate_sales(data=sales_data, filtered_data=filtered_sales)
        # display chart and table of recent sales
        display_sales_history(data=sales_data, monthly_data=monthly_sales)

    end_time = time.time()
    titles.caption("%s seconds to load" % round(end_time - start_time,2))

if __name__ == "__main__":
    main()
