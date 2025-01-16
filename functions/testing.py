import pandas as pd
from functions import *
from pykrige_test import *
spark = SparkSession.builder \
    .config('spark.driver.host',"localhost") \
    .appName("housing_data") \
    .getOrCreate()


def main():
    # Start declaring streamlit app content
    # Set page name
    st.set_page_config(layout="wide")

    # record set up time
    start_time = time.time()

    # load data
    sales_data = load_data(file = "data/sales_2021_on_geo.csv", _spark = spark, use_spark = True, add_h3 = True, lat_col="lat", lng_col="lng")
    st.caption("Average price per square foot by month!")
    sales_data = code_dates(sales_data)
    sales_data = filter_sales_data(sales_data,
                                   #date_min=int(date_min.replace("-", "")),
        )
    sales_data = prepare_sales_data(sales_data)    #adds price per sqft
    sales_monthly = sales_data.groupBy("year_month") \
                            .agg(
                                F.expr('percentile_approx(price_per_sqft, 0.5)').alias('price_per_sqft'),
                                F.count("sqft").alias("observations")) 
    # now add adjustment, to make all prices adjusted to one date. e.g. latest, get maximum month
    latest_month = sales_monthly.select("year_month").rdd.max()[0]
    numerator = sales_monthly.filter(F.col('year_month')==latest_month).select('price_per_sqft').collect()[0][0]
    # create price adjustment multiplier
    sales_monthly = sales_monthly.withColumn("adjustment", numerator/F.col('price_per_sqft'))
    sales_data = sales_data.join(sales_monthly.select("year_month","adjustment"), on="year_month",how="inner")
    sales_data = sales_data.withColumn('ppsf_adj', F.col("price_per_sqft")*F.col("adjustment"))
    st.caption("Prices adjusted to month : "+str(latest_month))
    #filtered_data = sales_data.select("ppsf_adj","lat","lng").toPandas().head(100)
    sales_data.toPandas().to_csv('outputs/sales_data.csv')
    # filtered_data.to_csv('outputs/filtered_data.csv')
    z = krige(sales_data)
    st.table(z)
    

if __name__ == "__main__":
    main()