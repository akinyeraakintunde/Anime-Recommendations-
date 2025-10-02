# train.py
import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

# Spark session setup
conf = SparkConf().set("spark.ui.port", "4050")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Load datasets
anime = spark.read.csv('anime.csv', header=True)
rating = spark.read.csv('rating.csv', header=True)

# Rename columns to avoid ambiguity
anime = anime.selectExpr('anime_id as anime_id', 'name as name', 'genre as genre', 'type as type','episodes as episodes', 'rating as anime_rating','members as members')
rating = rating.selectExpr('user_id as user_id', 'anime_id as anime_id' ,'rating as user_rating')

# Join datasets
data = anime.join(rating, on='anime_id', how='inner')

# Prepare data for ALS
data2 = data.selectExpr('user_id','name','user_rating')
data2 = data2.withColumn('user_rating',data['user_rating'].cast('int'))
data2 = data2.withColumn('user_id',data['user_id'].cast('int'))
data2 = data2.groupBy('name','user_id').avg('user_rating')

indexer = StringIndexer(inputCol="name", outputCol="names") 
indexer = indexer.fit(data2) 
data2 = indexer.transform(data2)

# Split dataset
(training, test) = data2.randomSplit([0.8, 0.2])

# Train ALS model
als = ALS(maxIter=5, implicitPrefs=True, userCol="user_id", itemCol="names", ratingCol='avg(user_rating)', coldStartStrategy="drop")
model = als.fit(training)

# Predict
predictions = model.transform(test)
predictions.show()

# Recommendations function
name_labels = indexer.labels
id_name = {x:y for x,y in enumerate(list(name_labels))}

def recommendedAnime(userId, limit=3):
    test =  model.recommendForAllUsers(limit).filter(col('user_id')==userId).select("recommendations").collect()
    topanime = []
    for item in test[0][0]:
        topanime.append(id_name[item.names])
    return topanime

# Show recommendations for 10 random users
x=0
while x<10:
    id_to_predict =  np.random.randint(1000)
    recommended = recommendedAnime(id_to_predict)
    print('*'*100)
    print(f"Recommended anime for user {id_to_predict}:")
    print(recommended)
    print('*'*100)
    x += 1
