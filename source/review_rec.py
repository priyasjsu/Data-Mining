import sklearn
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import sys
import requests
# from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')

#RestaurantReviewRecommendation Class for Restaurant recommendation this class returns result on review based trained model
class RestaurantReviewRecommendation:
    def __init__(self, business_data_file, model_file):
        self.df_business = None
        self.df = None
        self.P = None
        self.Q = None
        self.userid_vectorizer = None
        self.business_data_file = business_data_file
        self.model_file = model_file

    # Loading the data from csv file
    def load_data(self):
        self.df_business = pd.read_csv(self.business_data_file)

    # load the trained model from pickle file
    def predict(self, query):
        file = open(self.model_file, 'rb')
        self.P = pickle.load(file)
        self.Q = pickle.load(file)
        self.userid_vectorizer = pickle.load(file)
        self.predict_restaurant(query)

    # Cleaning The text
    @staticmethod
    def cleaning_text_process(text):
        stop_words = []
        if isinstance(text, str):
            text = ' '.join([str(word) for word in text.split()])
            text = ''.join([char for char in text if char not in string.punctuation])
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            filtered_text = ' '.join(filtered_tokens)
        else:
            filtered_text = ''
        return filtered_text
    # Cleaning the given query and recommending the restaurant based on the feature mentioned in query
    def predict_restaurant(self, query):
        test_df = pd.DataFrame([query], columns=['text'])

        #cleaning the query
        test_df['text'] = test_df['text'].apply(self.cleaning_text_process)
        #Transforming into vector
        test_vectors = self.userid_vectorizer.transform(test_df['text'])
        test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=self.userid_vectorizer.vocabulary_)
        predictItemRating = pd.DataFrame(np.dot(test_v_df.loc[0], self.Q.T), index=self.Q.index, columns=['Rating'])
        topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:7]
        name = []
        categories = []
        rating = []
        review_count = []
        print("7 Restaurant Recommended for given query = ", query)
        for i in topRecommendations.index:
            name.append(self.df_business[self.df_business['business_id'] == i]['name'].iloc[0])
            print(self.df_business[self.df_business['business_id'] == i]['name'].iloc[0])
            categories.append(self.df_business[self.df_business['business_id'] == i]['categories'].iloc[0])
            print(self.df_business[self.df_business['business_id'] == i]['categories'].iloc[0])
            rating.append(str(self.df_business[self.df_business['business_id'] == i]['stars'].iloc[0]))
            review_count.append(str(self.df_business[self.df_business['business_id'] == i]['review_count'].iloc[0]))
            print(
                str(self.df_business[self.df_business['business_id'] == i]['stars'].iloc[0]) + ' ' + str(
                    self.df_business[self.df_business['business_id'] == i]['review_count'].iloc[0]))

# get cmd line argument
arguments = sys.argv
query = arguments[1]

business_file = '../data/business_restaurant.csv'
model_file = '../data/review_model.pickle'

# Providing the path of data files
my_path = os.path.abspath(os.path.dirname(__file__))
business_file_path = os.path.join(my_path, business_file)

# Trained model file
model_file_path = os.path.join(my_path, model_file)

# Test data is loaded
df = pd.read_csv(business_file_path)
print("Sample of data", df.head(2))

#Initializing the class object to access RestaurantReviewRecommendation function
predictor = RestaurantReviewRecommendation(business_file_path, model_file_path)
predictor.load_data()
predictor.predict(query)
