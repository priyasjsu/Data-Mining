from setuptools import setup, Command
import zipfile
import os
import requests
import pickle

class ExtractCommand(Command):
    description = 'Extract the data zip file'
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        zip_url = "https://drive.google.com/uc?export=download&id=1y5yCp5NzZuqaRf7Ck4DNULGAPU9nsazy"
        restaurant_review_data = "https://drive.google.com/file/d/1pY6FD1y-9dLhuKJJEDOBlLQmrPjK-SVv/view?usp=sharing"
        # df1 = pd.read_csv(restaurant_file_path)
        business_data_url = "https://drive.google.com/uc?export=download&id=1V-cj0sZZlvJ4rJCu6mb8bYv4-Xd34vBp"

        model_file_url = "https://drive.google.com/uc?export=download&id=1WgnC9NBOeNHwBSauMliSHHRj8nkzP23M"

        response1 = requests.get(business_data_url)
        response2 = requests.get(restaurant_review_data)
        model_response = requests.get(model_file_url)

        business_data_path = "business_data.csv"
        cleaned_data_path = "restaurant_data.csv"
        model_path = "model.pkl"

        my_path = os.path.abspath(os.path.dirname(__file__))
        business_data_path = os.path.join(my_path, business_data_path)
        cleaned_data_path = os.path.join(my_path, cleaned_data_path)
        model_path = os.path.join(my_path, model_path)

        # Save the file locally
        with open(business_data_path, "wb") as file:
            file.write(response1.content.decode('utf-8'))

        with open(cleaned_data_path, "wb") as file1:
            file1.write(response2.content.decode('utf-8'))

        with open(model_path, "wb") as file2:
            pickle.dump(model_response.content, file2)
        # Check if the file is downloaded
        # while response.status_code != 200:
        #     time.sleep(1)  # Pause for 1 second

        # # Download the zip file
        # response = requests.get(zip_url)
        # zip_file_path = "temp.zip"  # Path to temporarily save the downloaded zip file
        #

        # with open(data_path, "wb") as file:
        #     file.write(response.content)
        # with zipfile.ZipFile(data_path, 'w') as zip_file:
        #     zip_file.writestr("temp1.zip", response.content)
        #
        # dest_path = './'
        # dest_path = os.path.join(my_path, dest_path)

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
