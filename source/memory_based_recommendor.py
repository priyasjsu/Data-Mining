import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gdown import download as drive_download
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


google_drive_paths = {
    "business_PA.csv": "https://drive.google.com/uc?id=1AKT1CsWrqQUnqkpQr_-T3N0SblsEjyPR",
    "review_PA.csv": "https://drive.google.com/uc?id=1wNXiz6_O4KYhltqY49ST9upB_af3m_xu",
}

business_path = 'data/business_PA.csv'
review_path = 'data/review_PA.csv'

def download_data(download_all=True, file=''):
    """
    Downloads the necessary data files.

    Parameters:
        download_all (bool, optional): If True, download all data files. If False, download a specific file specified by the 'file' parameter. Defaults to True.
        file (str, optional): The name of the specific file to download. This parameter is used when 'download_all' is False. Defaults to ''.

    """
    # if not os.path.isdir('data'):
    #     os.makedirs('data')
    # 
    # if download_all:
    #     for nn in google_drive_paths:
    #         url = google_drive_paths[nn]
    #         networkfile = os.path.join('data', nn)
    #         if not os.path.exists(networkfile):
    #             drive_download(url, networkfile, quiet=False)
    # else:
    #     url = google_drive_paths[file]
    #     networkfile = os.path.join('data', file)
    #     if not os.path.exists(networkfile):
    #         drive_download(url, networkfile, quiet=False)


class RecommenderSystem2:
    """
    A recommender system class that filters users and businesses based on a minimum number of reviews.

    Attributes:
        min_user_reviews (int): The minimum number of reviews a user must have given to be included in the analysis.
        data_business (pd.DataFrame): A DataFrame containing information about the businesses.
        top_10 (np.ndarray): The top 10 popular businesses based on the weighted rating.
        min_business_reviews (int): The minimum number of reviews a business must have received to be included in the analysis.
        filtered_data_review (pd.DataFrame): A DataFrame containing the filtered reviews based on the specified minimum number of user and business reviews.
        train (pd.DataFrame): The training set of the filtered_data_review after splitting.
        test (pd.DataFrame): The test set of the filtered_data_review after splitting.
    """
    def __init__(self, data_review, data_business, min_user_reviews=10, min_business_reviews=10):
        self.min_user_reviews = min_user_reviews
        self.data_business = data_business
        self.top_10 = self.get_top10_businesses(data_business)
        self.min_business_reviews = min_business_reviews
        self.filtered_data_review = self.filter_data(data_review)
        self.train, self.test = train_test_split(self.filtered_data_review, test_size=0.2, random_state=42)

    def filter_data(self, data_review):
        """
        Filters the data_review DataFrame based on the specified minimum number of user and business reviews.

        Parameters:
            data_review (pd.DataFrame): A DataFrame containing the reviews data.

        Returns:
            filtered_data_review (pd.DataFrame): A DataFrame containing the filtered reviews.
        """
        filtered_users = data_review['user_id'].value_counts()
        filtered_users = filtered_users[filtered_users >= self.min_user_reviews].index.tolist()

        filtered_businesses = data_review['business_id'].value_counts()
        filtered_businesses = filtered_businesses[filtered_businesses >= self.min_business_reviews].index.tolist()

        filtered_data_review = data_review[data_review['user_id'].isin(filtered_users)]
        filtered_data_review = filtered_data_review[filtered_data_review['business_id'].isin(filtered_businesses)]
        return filtered_data_review

    def weighted_rating(self, x):
        """
        Calculates the weighted rating for a business.

        Parameters:
            x (pandas.Series): A series containing the relevant data for a business, including 'stars' and 'review_count'.

        Returns:
            float: The weighted rating for the business.
        """
        C = self.data_business['stars'].mean()
        m = self.data_business['review_count'].quantile(0.90)
        R = x['stars']
        v = x['review_count']
        return (v / (v + m)) * R + (m / (v + m)) * C

    def get_business_names(self, bus_ids):
        """
        Returns the names of businesses given their IDs.

        Parameters:
            bus_ids (list): A list of business IDs.

        Returns:
            numpy.ndarray: An array containing the names of the businesses.
        """
        return (self.data_business[self.data_business['business_id'].isin(bus_ids)]['name']).values

    def similar_users_memory(self, user_id, metric='cosine', k=10):
        """
        Finds similar users based on their ratings using memory-based collaborative filtering.

        Parameters:
            user_id (int): The ID of the user for whom similar users are being calculated.
            metric (str, optional): The distance metric used for calculating user similarity. Defaults to 'cosine'.
            k (int, optional): The number of similar users to return. Defaults to 10.

        Returns:
            list: A list of user IDs representing the similar users.

        """
        user_ratings = self.train.pivot_table(index='user_id', columns='business_id', values='stars')
        user_ratings.fillna(0, inplace=True)
        
        try:
            user_index = np.where(user_ratings.index == user_id)[0][0]
            user_vector = user_ratings.iloc[user_index].values

            distances = []
            for i, row in enumerate(user_ratings.values):
                if i != user_index:
                    distances.append((user_ratings.index[i], 1 - cosine(user_vector, row)))
            
            distances = sorted(distances, key=lambda x: x[1], reverse=False)
            
            return [x[0] for x in distances[:k]]
        except IndexError:
            return []

    def recommend_business_memory(self, user_id, k=10):
        """
        Recommends businesses to a user based on memory-based collaborative filtering.

        Parameters:
            user_id (int): The ID of the user for whom the recommendations are being made.
            k (int, optional): The number of businesses to recommend. Defaults to 10.

        Returns:
            numpy.ndarray: An array containing the names of the recommended businesses.
        """
        similar_users_list = self.similar_users_memory(user_id, k=k)
        similar_users_data = self.train[self.train['user_id'].isin(similar_users_list)]
        grouped_data = similar_users_data.groupby('business_id')['stars'].mean().reset_index()
        sorted_data = grouped_data.sort_values(by='stars', ascending=False)
        if len(sorted_data) > 0:
            return self.get_business_names(sorted_data['business_id'].head(k).tolist())
        else:
            return self.get_business_names(self.top_10)

    def get_top10_businesses(self, data_business):
        """
        Returns the top 10 popular businesses based on the weighted rating.

        Parameters:
            data_business (pandas.DataFrame): The business data.

        Returns:
            list: A list of business IDs representing the top 10 popular businesses.
        """
        m = data_business['review_count'].quantile(0.90)
        qualified_restaurants = data_business[data_business['review_count'] >= m]
        qualified_restaurants['weighted_rating'] = qualified_restaurants.apply(lambda x: self.weighted_rating(x), axis=1)
        
        top_10_popular_restaurants = qualified_restaurants.sort_values('weighted_rating', ascending=False).head(10)
        return top_10_popular_restaurants['business_id'].tolist()


    # Function to predict the rating a user would give to a business
    def predict_rating_user_based(self, user_id, business_id, data, k=10):
        """
        Predicts the rating a user would give to a business based on user-based collaborative filtering.

        Parameters:
            user_id (int): The ID of the user for whom the rating is being predicted.
            business_id (int): The ID of the business for which the rating is being predicted.
            data (pandas.DataFrame): The data used for prediction.
            k (int, optional): The number of similar users to consider. Defaults to 10.

        Returns:
            float or None: The predicted rating of the user for the business, or None if no prediction can be made.
        """
        try:
            similar_users_list = similar_users_memory(user_id, data, k=k)
            similar_users_data = data[data['user_id'].isin(similar_users_list)]
            business_ratings = similar_users_data[similar_users_data['business_id'] == business_id]
        except IndexError:
            return None
        
        if business_ratings.empty:
            return None
        else:
            return business_ratings['stars'].mean()

    # Function to compute RMSE
    def compute_rmse_user_based(self, k=10):
        """
        Computes the Root Mean Square Error (RMSE) for user-based collaborative filtering.

        Parameters:
            k (int, optional): The number of similar users to consider. Defaults to 10.

        Returns:
            float: The computed RMSE.
        """
        train_data = self.train
        test_data = self.test.sample(1000, random_state=676)
        predictions = []
        actuals = []

        important_users = train_data['user_id'].value_counts()
        important_users = important_users[important_users >= 100].index.tolist()

        important_businesses = train_data['business_id'].value_counts()
        important_businesses = important_businesses[important_businesses >= 100].index.tolist()

        important_data = train_data[train_data['user_id'].isin(important_users)]
        important_data = important_data[important_data['business_id'].isin(important_businesses)]

        for _, row in test_data.iterrows():
            prediction = self.predict_rating_user_based(row['user_id'], row['business_id'], important_data, k=k)
            if prediction is not None:
                predictions.append(prediction)
                actuals.append(row['stars'])

        mse = mean_squared_error(actuals, predictions)
        rmse = sqrt(mse)

        return rmse

def main():
    print("Welcome to the Restaurant Recommender!")
    download_data()
    import os
    my_path = os.path.abspath(os.path.dirname(__file__))
    file_path1 = os.path.join(my_path, '../data/review_PA.csv')


    data_review = pd.read_csv(file_path1)

    file_path2 = os.path.join(my_path, '../data/business_PA.csv')

    data_business = pd.read_csv(file_path2)

    recommender2 = RecommenderSystem2(data_review, data_business)

    while True:
        print("\nPlease choose an option:")
        print("1. Recommend restaurants for a user")
        print("2. Find RMSE for User-based Recommendation system (WARNING: This will take about 2 minutes)")
        print("3. Exit")

        choice = input("Enter the number corresponding to your choice: ")

        if choice == '1':
            recommend_for_user_memory(recommender2)
        elif choice == '2':
            find_rmse_user_memory(recommender2)
        elif choice == "3":
            print("Thank you for using the Restaurant Recommender. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def recommend_for_user_memory(recommender2):
    """
    Recommends businesses (restaurants) to the user based on user interaction for memory-based collaborative filtering.

    Parameters:
        recommender2: The recommender object.

    """
    print("\nRecommend restaurant for a user")

    while True:
        print("\nPlease choose an option:")
        print("1. Select from sample user IDs")
        print("2. Enter a user ID manually")
        print("3. Go back to the main menu")

        choice = input("Enter the number corresponding to your choice: ")

        if choice == "1":
            sample_user_ids = ["_7bHUi9Uuf5__HHc_Q8guQ", "mh_-eMZ6K5RLWhZyISBhwA", "Dd1jQj7S-BFGqRbApFzCFw"]
            print("\nSample user IDs:")
            for idx, user_id in enumerate(sample_user_ids):
                print(f"{idx + 1}. {user_id}")

            user_id_choice = int(input("Enter the number corresponding to the user ID: "))
            user_id = sample_user_ids[user_id_choice - 1]
        elif choice == "2":
            user_id = input("Enter the user ID: ")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        recommended_business_names = recommender2.recommend_business_memory(user_id, k=10)
        print(f"\nRecommended businesses for user {user_id}:")
        for business_name in recommended_business_names:
            print(f"- {business_name}")

# Function to find similar users
def similar_users_memory(user_id, data, metric='cosine', k=10):
    """
    Finds similar users based on their ratings using memory-based collaborative filtering.

    Parameters:
        user_id (str): The ID of the user for whom similar users are being calculated.
        data (pandas.DataFrame): The data used for calculating similarity.
        metric (str, optional): The distance metric used for calculating user similarity. Defaults to 'cosine'.
        k (int, optional): The number of similar users to return. Defaults to 10.

    Returns:
        list: A list of user IDs representing the similar users.

    """
    user_ratings = data.pivot_table(index='user_id', columns='business_id', values='stars')
    #print(user_ratings.head())
    user_ratings.fillna(0, inplace=True)
    
    user_index = np.where(user_ratings.index == user_id)[0][0]
    user_vector = user_ratings.iloc[user_index].values

    distances = []
    for i, row in enumerate(user_ratings.values):
        if i != user_index:
            distances.append((user_ratings.index[i], 1 - cosine(user_vector, row)))
    
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in distances[:k]]


def find_rmse_user_memory(recommender2):
    """
    Computes the Root Mean Square Error (RMSE) on the test set for user-based collaborative filtering.

    Parameters:
        recommender2: The recommender object.

    """
    rmse_user_based = recommender2.compute_rmse_user_based()
    print("RMSE on test set for user-based collaborative filtering:", rmse_user_based)


if __name__ == "__main__":
    main()