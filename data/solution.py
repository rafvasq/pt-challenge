from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


def setup_data():
    weather_data = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("ignoreLeadingWhiteSpace", True) \
        .option("ignoreTrailingWhiteSpace", True) \
        .csv("/tmp/data/2019/*.gz") \
        .withColumnRenamed("STN---", "STN_NO")

    country_list = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("ignoreLeadingWhiteSpace", True) \
        .option("ignoreTrailingWhiteSpace", True) \
        .csv("/tmp/data/countrylist.csv")

    station_list = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("ignoreLeadingWhiteSpace", True) \
        .option("ignoreTrailingWhiteSpace", True) \
        .csv("/tmp/data/stationlist.csv")

    stations = station_list.join(country_list, on='COUNTRY_ABBR')

    station_weather_data = stations.join(weather_data, on='STN_NO')

    return station_weather_data


def get_hottest_country(station_weather_data):
    mean_temperatures = station_weather_data.select(["COUNTRY_FULL", "TEMP"]).where("TEMP < 9999.9").groupby(
        "COUNTRY_FULL") \
        .agg(F.mean("TEMP").alias("MEAN_TEMP"))

    mean_temperatures.orderBy(F.desc("MEAN_TEMP")).take(1)

    first_row = mean_temperatures.orderBy(F.desc("MEAN_TEMP")).take(1)[0]
    country = first_row[0]
    value = first_row[1]

    return country, value


def get_most_tornados(station_weather_data):
    w1 = Window.partitionBy("COUNTRY_FULL").orderBy(["COUNTRY_FULL", "YEARMODA"])
    w2 = Window.partitionBy("DIFF").orderBy("COUNTRY_FULL")

    tornado_data = station_weather_data.select(['COUNTRY_FULL', "YEARMODA", "FRSHTT"]).where("FRSHTT == '10011'")

    tornado_data = tornado_data.withColumn("PREV", F.lag(tornado_data.YEARMODA).over(w1))

    tornado_data = tornado_data.withColumn("DIFF", F.when(F.isnull(tornado_data.YEARMODA - tornado_data.PREV), 0)
                                           .otherwise(tornado_data.YEARMODA - tornado_data.PREV))

    tornado_data = tornado_data \
        .withColumn("GRP", F.row_number().over(w1) - F.row_number().over(w2)) \
        .withColumn("STREAK", F.row_number().over(Window.partitionBy("GRP").orderBy(["COUNTRY_FULL", "YEARMODA"])))

    first_row = tornado_data.orderBy(F.desc("STREAK")).take(1)[0]
    country = first_row[0]
    value = first_row[-1]

    return country, value


def get_most_windy(station_weather_data):
    mean_winds = station_weather_data.select(["COUNTRY_FULL", "WDSP"]).where("TEMP < 999.9") \
        .groupby("COUNTRY_FULL") \
        .agg(F.mean("WDSP").alias("MEAN_WIND"))
    second_row = mean_winds.orderBy(F.desc("MEAN_WIND")).take(2)[1]
    country = second_row[0]
    value = second_row[1]

    return country, value


if __name__ == "__main__":
    spark = SparkSession.builder.appName("EQWorks").getOrCreate()

    station_weather_data = setup_data()

    print(get_hottest_country(station_weather_data))

    print(get_most_tornados(station_weather_data))

    print(get_most_windy(station_weather_data))

    spark.stop()
