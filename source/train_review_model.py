import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk
import pickle

nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class ReviewBasedModel:
    def __init__(self, business_data_file, restaurant_data_file, model_file):
        self.business_data_file = business_data_file
        self.restaurant_data_file = restaurant_data_file
        self.model_file = model_file
        self.df_business = None
        self.df = None
        self.P = None
        self.Q = None
        self.userid_vectorizer = None

    def load_data(self):
        self.df_business = pd.read_csv(self.business_data_file)
        self.df = pd.read_csv(self.restaurant_data_file)

    @staticmethod
    def cleaning_text_process(text):
        stop_words = []

        for word in stopwords.words('english'):
            s = [char for char in word if char not in string.punctuation]
            stop_words.append(''.join(s))
        filtered_text = ''
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

    @staticmethod
    def matrix_factorization(R, P, Q, steps=20, gamma=0.001, lamda=0.02):
        for step in range(steps):
            for i in R.index:
                for j in R.columns:
                    if R.loc[i, j] > 0:
                        eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                        P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                        Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
            e = 0
            for i in R.index:
                for j in R.columns:
                    if R.loc[i, j] > 0:
                        e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                                    pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
            if e < 0.001:
                break
        return P, Q

        # Store P, Q, and vectorizer in a pickle file
        def save_trained_model(self, P, Q, userid_vectorizer):
            output = open(
                '../yelp_recommendation_model_8.pkl_2', 'wb')
            pickle.dump(P, output)
            pickle.dump(Q, output)
            pickle.dump(userid_vectorizer, output)
            output.close()

        def clean_text_process(self, text):
            stop_words = []
            filtered_text = ''
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

        def extract_feature_tfidf(self, userid_df, restaurant_df):
            # Apply TFIDF Vectorizer to extract the features from the text
            # userid vectorizer
            userid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=2000)
            userid_vectors = userid_vectorizer.fit_transform(userid_df['review'])

            # Restaurant id vectorizer
            restaurantid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=2000)
            restaurantid_vectors = restaurantid_vectorizer.fit_transform(restaurant_df['review'])
            return userid_vectors, restaurantid_vectors, userid_vectorizer, restaurantid_vectorizer

        def reviews_based_model_train(self):
            # df = df.drop(df.index[30:345000])
            # data['review'] = data['review'].apply(cleaning_text_process)
            data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data-mining-rest-rec/cleaned_review.csv")
            userid_df = data[['user_id', 'review']]
            restaurant_df = data[['restaurant_id', 'review']]

            # Aggregating the reviews based on restaurant_id and user_id
            userid_df = userid_df.groupby('user_id').agg({'review': ' '.join})
            restaurant_df = restaurant_df.groupby('restaurant_id').agg({'review': ' '.join})
            userid_vectors, restaurantid_vectors, userid_vectorizer, restaurantid_vectorizer = self.extract_feature_tfidf(
                userid_df, restaurant_df)

            # create a matrix of users and restaurant with the ratings
            # user_counts = data['user_id'].value_counts()
            # data_filtered = data[data['user_id'].isin(user_counts[user_counts >= 5].index)]
            userid_rating_matrix = pd.pivot_table(data, values='rating', index=['user_id'], columns=['restaurant_id'])
            P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index,
                             columns=userid_vectorizer.vocabulary_.keys())
            Q = pd.DataFrame(restaurantid_vectors.toarray(), index=restaurant_df.index,
                             columns=restaurantid_vectorizer.vocabulary_.keys())
            P, Q = self.matrix_factorization(userid_rating_matrix, P, Q, steps=20, gamma=0.001, lamda=0.02)
            self.save_trained_model(P, Q, userid_vectorizer)

    # Usage:
    df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data-mining-rest-rec/restaurant_data.csv')
    model = ReviewBasedModel(df)
    model.reviews_based_model_train()