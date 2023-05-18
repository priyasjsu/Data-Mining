from recommender import HotelRecommender
import pandas as pd
from review_classification import Sentiment
from gdrive_downloader import g_downloader
import os


print("Downloading Review Data")
g_downloader('18cBwZqZTmcX1vBD-zqI-86O0HKdOYRKF')
print("Downloaded Review Data")
print("loading Review Dataframe")
df_review = pd.read_csv('business_review_pa.csv')
print("Review Dataframe Loaded")
sent = Sentiment()
s = HotelRecommender(df_review, sent, 'business_restaurant.csv')
rel = []
while True:
    user_input = input("Enter a business name to get recommendations or press 'q' to exit: ")
    if user_input == 'q':
        break
    
    _ = s.recommend(user_input, 5)
    if _ is not -1:
        valid = input("Please enter the number of recommendations which you find relevant: ")
        rel.append(int(valid))
        print("Precision is: ", round((int(valid)/5),2))

map = 0
for i in rel:
    map+=i/5
map=map/len(rel)
print("Mean Average Precision is: ", round(map,2))

