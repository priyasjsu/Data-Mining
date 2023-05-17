from setuptools import setup, Command
import zipfile
import os
import requests
import pickle
import gdown

class ExtractCommand(Command):
    description = 'Extract the data zip file'
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        # data_path = "./data.zip"
        # dest_path = './'
        review_file_id = "1wNXiz6_O4KYhltqY49ST9upB_af3m_xu"
        business_PA_id = "129RGPPBDHrf7jTluorluaYoNPfoj2pnJ"
        business_restaurant_id = "1V-cj0sZZlvJ4rJCu6mb8bYv4-Xd34vBp"
        cleaned_review_id = "1Vb61c5g1W3QZ4z_77vHeXFFxHfClyht3"
        restaurant_data_id = "1pY6FD1y-9dLhuKJJEDOBlLQmrPjK-SVv"

        model_file_id = "1WgnC9NBOeNHwBSauMliSHHRj8nkzP23M"
        business_json = "1-CwSFO8efdwlLHLkA2jtsYJHcJHL3Xw3"

        file_url = f'https://drive.google.com/uc?id={review_file_id}'
        business_PA_url = f'https://drive.google.com/uc?id={business_PA_id}'
        business_restaurant_url = f'https://drive.google.com/uc?id={business_restaurant_id}'
        cleaned_review_url = f'https://drive.google.com/uc?id={cleaned_review_id}'
        restaurant_data_url = f'https://drive.google.com/uc?id={restaurant_data_id}'
        business_json_url = f'https://drive.google.com/uc?id={business_json}'
        model_url = f'https://drive.google.com/uc?id={model_file_id}'


        my_path = os.path.abspath(os.path.dirname(__file__))
        review_csv_PA = os.path.join(my_path, "./data/review_PA.csv")
        business_PA = os.path.join(my_path, "./data/business_PA.csv")
        business_restaurant = os.path.join(my_path, "./data/business_restaurant.csv")
        cleaned_restaurant = os.path.join(my_path, "./data/cleaned_review.csv")
        restaurant_data = os.path.join(my_path, "./data/restaurant_data.csv")
        business_json = os.path.join(my_path, "./data/business.json")
        model_pickle_review = os.path.join(my_path, "./data/model_pickle_review.pickle")
        
        os.makedirs('./data', exist_ok=True)
        # Retrieve the file content
        gdown.download(file_url, review_csv_PA, quiet=False)
        gdown.download(business_PA_url, business_PA,  quiet=False)
        gdown.download(business_restaurant_url, business_restaurant,  quiet=False)
        gdown.download(cleaned_review_url, cleaned_restaurant, quiet=False)
        gdown.download(restaurant_data_url, restaurant_data,  quiet=False)
        gdown.download(business_json_url, business_json,  quiet=False)
        gdown.download(model_url, model_pickle_review,  quiet=False)

        # zip_path = data_path  # Replace with the actual path to your zip file
        # extract_folder = dest_path  # Replace with the path where you want to extract the files
        #
        # # Extract the contents of the zip file
        # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #     zip_ref.extractall(extract_folder)
        # print("Zip file extracted successfully!")
        # # Remove the temporary zip file
        # os.remove(data_path)

setup(
    name='restaurant-recommendation',
    version='1.0',
    packages=['source', 'source.utils'],
    url='https://github.com/priyasjsu/Data-Mining.git',
    license='free',
    author='Priya Khandelwal',
    author_email='priyakhandelwal',
    description='A Data Mining project for restaurant recommendation using multiple technique',
    install_requires=[
        'requests==2.20.0'
    ],
    cmdclass={
        'extract': ExtractCommand,
    }
)
