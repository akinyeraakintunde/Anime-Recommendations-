# Anime Recommendation System

This project implements a collaborative filtering-based anime recommendation system using PySpark.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place `anime.csv` and `rating.csv` in the project directory.

## Run the Training Script

```bash
python train.py
```

This will train the ALS model and output recommended anime for 10 random users.

## Files

- `train.py`: Main script to train ALS and generate recommendations.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
