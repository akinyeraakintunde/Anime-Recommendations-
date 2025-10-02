# Anime Recommendation System

## Overview

This project is a **Collaborative Filtering-based Anime Recommendation System** built using **PySpark**. It leverages user ratings and anime metadata to provide personalized recommendations for users. The system can suggest the top anime for individual users based on past ratings using **ALS (Alternating Least Squares)**.

---

## Features

* Load and process large datasets with PySpark.
* Merge anime metadata (`anime.csv`) with user ratings (`rating.csv`).
* Explore datasets: top anime by rating, most episodes, and genre analysis.
* Build a collaborative filtering model using ALS.
* Generate top-N recommendations for individual users.

---

## Dataset

The project requires two datasets from Google Drive:

1. **anime.csv**
   Columns: `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members`

2. **rating.csv**
   Columns: `user_id`, `anime_id`, `rating`

**Note:** Negative ratings (-1) indicate that the user has not rated that anime.

---

## Installation

### Prerequisites

* Python 3.7+
* Java 8
* Google Colab (optional but recommended)
* Google Drive account (for dataset access)

### Install Required Packages

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
pyspark==3.2.0
numpy
pandas
matplotlib
PyDrive
```

---

## Setup

1. **Authenticate Google Drive** to access dataset files:

```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```

2. **Download datasets from Google Drive**:

```python
downloaded = drive.CreateFile({'id': 'YOUR_ANIME_CSV_ID'})
downloaded.GetContentFile('anime.csv')

downloaded = drive.CreateFile({'id': 'YOUR_RATING_CSV_ID'})
downloaded.GetContentFile('rating.csv')
```

---

## Usage

### 1. Initialize PySpark

```python
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

conf = SparkConf().set("spark.ui.port", "4050")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
```

### 2. Load Datasets

```python
anime = spark.read.csv('anime.csv', header=True)
rating = spark.read.csv('rating.csv', header=True)
```

### 3. Preprocess Datasets

```python
anime = anime.selectExpr('anime_id as anime_id', 'name as name', 'genre as genre', 'type as type','episodes as episodes','rating as anime_rating','members as members')
rating = rating.selectExpr('user_id as user_id', 'anime_id as anime_id' ,'rating as user_rating')
```

### 4. Merge Datasets

```python
data = anime.join(rating, on='anime_id', how='inner')
```

### 5. Exploratory Analysis

* Top 10 anime by user rating
* Top 10 anime by number of episodes
* Top genres based on user rating

### 6. Build Recommendation Model

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

data2 = data.selectExpr('user_id','name','user_rating')
data2 = data2.withColumn('user_rating',data2['user_rating'].cast('int'))
data2 = data2.withColumn('user_id',data2['user_id'].cast('int'))
data2 = data2.groupBy('name','user_id').avg('user_rating')

indexer = StringIndexer(inputCol="name", outputCol="names")
indexer = indexer.fit(data2)
data2 = indexer.transform(data2)

(training, test) = data2.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, implicitPrefs=True, userCol="user_id", itemCol="names", ratingCol='avg(user_rating)', coldStartStrategy="drop")
model = als.fit(training)
predictions = model.transform(test)
```

### 7. Get Recommendations

```python
name_labels = indexer.labels
id_name = {x:y for x,y in enumerate(list(name_labels))}

def recommendedAnime(userId, limit=3):
    test = model.recommendForAllUsers(limit).filter(col('user_id')==userId).select("recommendations").collect()
    topanime = [id_name[item.names] for item in test[0][0]]
    return topanime
```

---

## Example Output

```
User 364: ['Fullmetal Alchemist: Brotherhood', 'Death Note', 'Sword Art Online']
User 648: ['Shingeki no Kyojin', 'Death Note', 'Sword Art Online']
User 215: ['Death Note', 'Shingeki no Kyojin', 'Sword Art Online']
```

## References

* [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
* [Alternating Least Squares for Collaborative Filtering](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

## Author

* Developed for educational purposes by Ibrahim Akinyera
