from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType

def load_triplets(path):
    """Format dataframe schema and load triplets into it, returns the filed dataframe
    and the SparkSession to be used on future functions
    """
    spark = SparkSession.builder.appName("MSD").getOrCreate()
    schema = StructType() \
        .add("userId", StringType()) \
        .add("songId", StringType()) \
        .add("count", IntegerType()) 
    return spark, spark.read.csv(path, sep="\t", schema=schema)
