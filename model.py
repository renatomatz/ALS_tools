from pyspark.sql.functions import col
from pyspark.ml.ALS import ALS
from tester import roem_cross_val

def train_model(data, model):
    """This assumes you have tested models using reduced data and don't
    necessarily know what are the best params"""

    print("Model Parameters:")
    print("Rank: {}".format(model.getRank()))
    print("MaxIter: {}".format(model.getMaxIter()))
    print("RegParam: {}".format(model.getRegParam()))
    print("Aplpha: {}".format(model.getAlpha()))

    return model.fit(data)

def get_user_predictions(df, model, users=[], n=5):
    """Get top <n> recommendations for the requested users, if none are 
    provided, recommendations are generated for all data

    - <users> must be a list
    """

    assert isinstance(users, list)

    if len(users) > 0:
        df = df.filter(col("userId").isin(users))

    return df.group_by("userId").sort(col("prediction")).limit(n)
