from functions import *

APP_TITLE = "Housing search map app"
APP_SUB_TITLE = "Produced by Marie Thompson using streamlit. Data from Zillow, Foursquare API and King County Department of Assessments"

def main():

    # load data
    sales_data = pd.read_csv("data/sales_2021_on_geo.csv")
    sales_data = sales_data.h3.geo_to_h3(resolution=8, lat_col="lat", lng_col="lng")
    sales_data = sales_data.h3.h3_to_geo_boundary().reset_index()

    listings_data = pd.read_csv("data/all-listings.csv")
    # remove listings more than 10,000,000
    listings_data = listings_data[listings_data["price"] < 10000000]
    listings_data = listings_data.h3.geo_to_h3(
        resolution=8, lat_col="latitude", lng_col="longitude"
    )
    listings_data = listings_data.h3.h3_to_geo_boundary().reset_index()
    grouped_listings = aggregate_listings(listings_data)

    daycare_data = pd.read_csv("data/daycares.csv")

    # Start declaring streamlit app content
    # Set page name
    st.set_page_config(APP_TITLE, layout="wide")

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
    if location_chat:
        chat.write(f"User has sent the following prompt: {location_chat}")
        generate_response(
            location_chat,
            dataframes=[
                daycare_data.rename(
                    columns={
                        "geocodes.main.latitude": "latitude",
                        "geocodes.main.longitude": "longitude",
                    }
                ).loc[:, ["name", "latitude", "longitude"]],
                listings_data.loc[
                    :, ["price", "latitude", "longitude", "propertyType", "bedrooms"]
                ],
            ],
            #  dataframes = [daycare_data,listings_data.reset_index()],
            container=chat,
        )

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
        bounding_box = display_map(
            listings_data=listings_data,
            aggregate_listing_data=grouped_listings,
            places_data=daycare_data,
            places_name="Day cares",
        )
        # filter listings based on map bounds for listing metrics
        listings = filter_listings(data=listings_data, bounding_box=bounding_box)
        # filter sales based on map bounds and bedroom/date filter
        # filter data with bounding_box and dates/beds
        sales_data = prepare_sales_data(sales_data)
        # filter for 'This area' in sale price chart, and recent sales
        filtered_sales = sales_data[
            sales_data["lat"].between(bounding_box[0], bounding_box[2])
        ]
        filtered_sales = sales_data[
            sales_data["lng"].between(bounding_box[1], bounding_box[3])
        ]

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


if __name__ == "__main__":
    main()
