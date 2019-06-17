from pyspark.sql.functions import monotonically_increasing_id

def sub_by_int_id(df, col):
    """Substitute a string column by a unique integer id
    - Keep in mind that 
    - col must be a string and existent
    """
    new_col = "unique_" + col
    
    temp = df.select(col).distinct().coalesce(1) \
                .withColumn(new_col, monotonically_increasing_id()).persist() \
                .select(new_col, col)

    return df.join(temp, on=col, how="left").drop(col).withColumnRenamed(new_col, col)

def make_association_matrix(df, to_col="userId", to_row="songId", value="count"):
    """Pivot DataFrame and fill NAs with zeroes
    - df should have three columns
    - to_col should be the name of the column whose values will become distinct columns
    - to_row is the name of the colum
    """
    assert len(df.columns) == 3
    return df.groupBy(to_row).pivot(to_col).sum(value).fillna(0)

def make_implicit_rating_matrix(df):
    """As the ALS model requires implicit ratings, we convert a matrix containing only
    songs each user listened to a much longer matrix where all possible user-song combinations
    are present and 0s where a combination didn't originally exist
    """
    users = df.select("userId").distinct()
    songs = df.select("songId").distinct()

    return users.crossJoin(songs).join(df, on=["userId", "songId"], how="left").fillna(0)

def prepare_for_als(df, userCol="userId", songCol="songId"):
    """Prepare a raw dataframe for the ALS algorithm by making column ID's unique integers
    and making an implicit ratings matrix out of them
    """
    users = df.select(userCol).distinct().coalesce(1).withColumn("uniqueUsers", monotonically_increasing_id()).persist()
    songs = df.select(songCol).distinct().coalesce(1).withColumn("uniqueSongs", monotonically_increasing_id()).persist()

    return users.crossJoin(songs).join(df, on=[userCol, songCol], how="left") \
            .fillna(0).select("uniqueUsers", "uniqueSongs", "count") \
            .withColumnRenamed("uniqueUsers", userCol) \
            .withColumnRenamed("uniqueSongs", songCol)
