from pyspark.sql import SparkSession, functions as f#

spark = SparkSession.builder \
        .config('spark.driver.host',"localhost") \
        .appName("housing_data") \
        .getOrCreate()#
from functions import load_data
sales_data = load_data(file = "data/sales_2021_on_geo.csv", _spark = spark, add_h3 = True, use_spark = True, lat_col="lat", lng_col="lng")
cols = sales_data.columns
print(cols)
#test = spark.read.csv("data/sales_2021_on_geo.csv", header=True, inferSchema=True)
#cols = test.columns
#print(cols)