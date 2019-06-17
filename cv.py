from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from ROEM_function import ROEM

def roem_cross_val(spark, df, nfolds=5):
    """Manullay create a taining grid with different parameters for an ALS model, 
    train and evaluate the models (using ROEM) and return a dictionary containing:
    - models: a list of dicts for all models trained containing:
    -- model: the actual model
    -- cross_val: cross validation results
    -- ROEM: this model's ROEM score
    - best: the best model's index

    This proceedure is mostly done manually as the ROEM evaluation metric is not 
    yet supported by spyspark, I've used an implementation by @jamenlong and 
    adapted his cross-validation method to be more flexible

    folds > 1
    """
    assert nfolds > 1

    rank = [5, 10, 15]
    maxIter = [10, 20, 40]
    alpha = [20, 60, 80]
    regParams = [0.01, 0.05, 0.5, 1]

    models = []

    for r in rank:
        for m in maxIter:
            for a in alpha:
                for reg in regParams:
                    models.append(ALS(userCol="userId", 
                                      itemCol="songId", 
                                      ratingCol="count", 
                                      coldStartStrategy="drop", 
                                      nonnegative=True,
                                      implicitPrefs=True,
                                      rank=r,
                                      maxIter=m,
                                      alpha=a,
                                      regParam=reg))
    
    train, test = df.randomSplit([0.8, 0.2])

    train_splits = train.randomSplit([1/nfolds]*nfolds)

    folds = []

    for i in range(nfolds):
        temp = iter(train_splits[0:i] + train_splits[1+i:nfolds]) 
        # leave on split behind
        fold_u = next(temp)
        for fold in temp:
            fold_u = fold_u.union(fold)
            # join remaining splits
        folds.append((fold_u, train_splits[i]))
        # have remaining split as test data

    del train_splits

    results = {"models":[], "best":None}

    for i, model in enumerate(models):
        mod = {"model":model, "cv_results":[], "ROEM":None}

        for j, ft in enumerate(folds):
            # ft stands for fit-train
            predictions = model.fit(ft[0]).transform(ft[1])
            mod["cv_results"].append(ROEM(spark, predictions))
            print("Model {}, Fold {}: {}".format(i+1, j+1, mod["cv_results"][j]))

        mod["ROEM"] = ROEM(spark, model.fit(train).transform(test))

        results["models"].append(mod)

        if (mod["ROEM"] > 0) and ((not results["best"]) or (mod["ROEM"] < results["models"][results["best"]]["ROEM"])):
            results["best"] = i

        print("Model {}, FINAL RESULT: {}".format(i, mod["ROEM"]))

    return results