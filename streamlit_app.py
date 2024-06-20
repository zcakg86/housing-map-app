from functions import *

APP_TITLE = 'Housing search map app'
APP_SUB_TITLE = 'Produced by Marie Thompson using streamlit. Data from King County Assessors, Zillow and Foursquare API'

def main():

    st.set_page_config(APP_TITLE, layout = 'wide')
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    # load data
    sales_data = pd.read_csv('data/sales_2021_on_geo.csv')
    sales_data = sales_data.h3.geo_to_h3(resolution = 8, lat_col = 'lat', lng_col = 'lng')
    sales_data = sales_data.h3.h3_to_geo_boundary().reset_index()

    listings_data = pd.read_csv('data/all-listings.csv')
    listings_data = listings_data.h3.geo_to_h3(resolution = 8, lat_col = 'latitude', lng_col = 'longitude')
    listings_data = listings_data.h3.h3_to_geo_boundary().reset_index()
    grouped_listings = aggregate_listings(listings_data)

    daycare_data = pd.read_csv('data/daycares.csv')

    # display filters and chat in sidebar
    beds_min, beds_max = display_bedroom_filter(sales_data, st.sidebar)
    date_min, date_max = display_time_filters(sales_data,'sale_date', st.sidebar)

    chat = st.sidebar.container()
    location_chat = chat.chat_input(placeholder="Chat about listings")
    if location_chat:
        chat.write(f"User has sent the following prompt: {location_chat}")
        generate_response(location_chat,
                          dataframes = [daycare_data.rename(columns={'geocodes.main.latitude':'latitude',
                                                                     'geocodes.main.longitude':'longitude'})\
                                        .loc[:,['name','latitude','longitude']],
                                        listings_data.loc[:,['price','latitude','longitude','propertyType','bedrooms']]],
                        #  dataframes = [daycare_data,listings_data.reset_index()],                                  
                          container = chat)
        
    # Display metrics
    col_left, col_right = st.columns((2,1))
    with col_left:
        # create container
        container = st.container(border = False)
        # display map and return map bounds
        bounding_box = display_map(listings_data=listings_data, aggregate_listing_data=grouped_listings, places_data = daycare_data, places_name = 'Day cares')
        col1, col2 = container.columns(2)
        # filter listings based on map bounds and bedroom filter
        listings = filter_listings(data=listings_data, bounding_box=bounding_box, beds_min=beds_min, beds_max=beds_max)
        with col1:
            display_listings_aggregate(data=listings, field_name='price', metric='count', metric_title=f'Total listings')
        with col2:
            display_listings_aggregate(data=listings, field_name='price', metric='median', metric_title=f'Median listing price')

    with col_right: # presenting sales history data
        st.header("Sales history")
        # filter data with bounding_box and dates/beds 
        sales = prepare_sales_data(sales_data, date_min=int(date_min.replace('-', '')),
                                  date_max=int(date_max.replace('-', '')), beds_min=beds_min, beds_max=beds_max)
        filtered_sales = sales[sales['lat'].between(bounding_box[0],bounding_box[2])]
        filtered_sales = sales[sales['lng'].between(bounding_box[1], bounding_box[3])]
        # aggregate by month for displaying chart
        monthly_sales = aggregate_sales(data = sales, filtered_data = filtered_sales)  
        col3,col4 = st.columns(2)
        with col3:
            display_sales_aggregate(data=filtered_sales,
                                    field_name='sale_price', metric='count',
                                    metric_title=f'Total sales')
        with col4:
            display_sales_aggregate(data=filtered_sales, field_name='sale_price', metric='median',
                                    metric_title=f'Median sale price')
        display_sales_history(data=sales, monthly_data = monthly_sales)

if __name__ == "__main__":
    main()
