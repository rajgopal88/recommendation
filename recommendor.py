#!/usr/bin/env python

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.recommendation import ALS

#from pyspark import SparkConf, SparkContext 

def parsePurchases(line):
    """
    Parses a purchases record in Purchases format user_Id::sku_Id::purchases::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseSku(line):
    """
    Parses a sku record in Sku format skuId::sku_Title .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadPurchases(purchasesFile):
    """
    Load purchase from file.
    """
    if not isfile(purchasesFile):
        print "File %s does not exist." % purchasesFile
        sys.exit(1)
    f = open(purchasesFile, 'r')
    purchases = filter(lambda r: r[2] > 0, [parsePurchases(line)[1] for line in f])
    f.close()
    if not purchases:
        print "No purchases provided."
        sys.exit(1)
    else:
        return purchases

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    print "RESULT_data:%s " % ((data.map(lambda x: (x[0], x[1]))).take(50))
    predictions1 = model.predictAll(data.map(lambda x: (x[0], x[1])))
    print "RESULT1: %s" % predictions1
    predictionsAndRatings = predictions1.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    #print "RESULT2: %s" % predictions1.take(11)
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "recommendaion.py DataDir personalPurchasesFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("Reccomendation") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    #parseData()

    sqlContext = SQLContext(sc)

    #connect to database
    jdbcDF = sqlContext.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://localhost/dbservice?user=root&password=password") \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .option("dbtable", "viewedproduct") \
        .option("user", "root") \
        .option("password", "password") \
        .load()

    print(jdbcDF)
    j = jdbcDF.toDF('xc_sku', 'psid')

    PurchasesHomeDir = sys.argv[1]
    j.write.csv(sc.textFile(join(PurchasesHomeDir, "purchases.dat")), 'append')
    #csv_writer(jdbcDF, sc.textFile(join(PurchasesHomeDir, "purchases.dat")))


    # load personal purchases
    myPurchases = loadPurchases(sys.argv[2])
    myPurchasesRDD = sc.parallelize(myPurchases, 1)

    # print "RESULT: My purchases are here : %s " % (myPurchasesRDD.take(10))
    # load purchases and product titles
    
    # prchases is an RDD of (last digit of timestamp, (user_Id, sku_Id, purchases))
    purchases = sc.textFile(join(PurchasesHomeDir, "purchases.dat")).map(parsePurchases)

    # print "RESULT: The purchases are here : %s " % (purchases)

    # sku is an RDD of (sku_Id, sku_Title)
    sku = dict(sc.textFile(join(PurchasesHomeDir, "sku.dat")).map(parseSku).collect())

    # print "RESULT: The skus are here : %s " % (sku)

    numPurchases = purchases.count()
    numUsers = purchases.values().map(lambda r: r[0]).distinct().count()
    numSkus = purchases.values().map(lambda r: r[1]).distinct().count()

    print purchases.take(100)
    print "RESULT:   Got %d ratings from %d users on %d purchases." % (numPurchases, numUsers, numSkus)

    # split purchases into train (60%), validation (20%), and test (20%) based on the 
    # last digit of the timestamp, add myPurchases to train, and cache them

    # training, validation, test are all RDDs of (userId, skuId, rating)
    #need to work on this later once data is clear

    numPartitions = 4
    training = purchases.map(lambda x: x[1]).filter(lambda x: x[0] < 6) \
      .union(myPurchasesRDD) \
      .repartition(numPartitions) \
      .cache()

    validation = purchases.map(lambda x: x[1]).filter(lambda x: x[0] >= 6 and x[0] < 10) \
      .repartition(numPartitions) \
      .cache()

    test = purchases.map(lambda x: x[1]).filter(lambda x: x[0] >= 10).cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "RESULT: Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # # train models and evaluate them on the validation set

    ranks = [4, 8]
    lambdas = [0.1, 10.0]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        print "RESULT_MODEL: %s" % (model)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RESULT:   RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # compare the best model with a naive baseline that always returns the mean rating
    meanPurchase = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanPurchase - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."

    # # make personalized recommendations

    myPurchasedSkuIds = set([x[1] for x in myPurchases])
    candidates = sc.parallelize([m for m in sku if m not in myPurchasedSkuIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print "Products recommended for you:"
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, sku[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()
