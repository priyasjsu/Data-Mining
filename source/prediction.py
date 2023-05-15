import sklearn
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
# from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')

class RestaurantPrediction:
    def __init__(self, business_data_file, model_file):
        self.df_business = None
        self.df = None
        self.P = None
        self.Q = None
        self.userid_vectorizer = None
        self.business_data_file = business_data_file
        # self.restaurant_data_file = restaurant_data_file
        self.model_file = model_file

    def load_data(self):
        self.df_business = pd.read_csv(self.business_data_file)
        # self.df = pd.read_csv(self.restaurant_data_file)

    def predict(self, query):
        with open(self.model_file, 'rb') as file:
            self.P = pickle.load(file)
            self.Q = pickle.load(file)
            print('self.Q', self.Q)
            self.userid_vectorizer = pickle.load(file)
        self.predict_restaurant(query)

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

    def predict_restaurant(self, query):
        test_df = pd.DataFrame([query], columns=['text'])
        test_df['text'] = test_df['text'].apply(self.cleaning_text_process)

        test_vectors = self.userid_vectorizer.transform(test_df['text'])
        test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=self.userid_vectorizer.vocabulary_)

        predictItemRating = pd.DataFrame(np.dot(test_v_df.loc[0], self.Q.T), index=self.Q.index, columns=['Rating'])
        topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:7]
        print('predictItemRating', predictItemRating)
        name = []
        categories = []
        rating = []
        review_count = []

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
            print('')

business_file = '../business_restaurant.csv'
restaurant_file = '../restaurant_data.csv'
model_file = '../yelp_recommendation_model_8.pkl_1'

my_path = os.path.abspath(os.path.dirname(__file__))
business_file_path = os.path.join(my_path, business_file)
restaurant_file_path = os.path.join(my_path, restaurant_file)
model_file_path =  os.path.join(my_path, model_file)

query = "i want to go for vegetarian place"
df = pd.read_csv(business_file_path)
# df1 = pd.read_csv(restaurant_file_path)

print(df.head(2))

predictor = RestaurantPrediction(business_file_path, model_file_path)
predictor.load_data()
predictor.predict(query)
