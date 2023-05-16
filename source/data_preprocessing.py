import json
import csv
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import folium
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import requests
# from io import StringIO
import os
import io



class DataConverter:
    def __init__(self, path):
        # self.csv_file = csv
        self.my_path = path

    # create csv from json
    def convert_json_to_csv(self, path, file_name):
        # since pd.readjson will cause memory error, we read the file line by line
        review = []
        with open(path, encoding='utf-8') as fin:
            i = 0
            for line in fin:
                line_contents = json.loads(line)
                review.append(line_contents)
        review = pd.DataFrame(review)
        review.shape
        # WRITE THE REVIEW FILE INTO CSV FORMAT
        review.to_csv(file_name)
        #We have filtered json and save in csv file we don't need to do it again.
        # convert_json_to_csv(review_json, "./data/reviews.csv")

    def open_json_file(file_path):
        # Open the JSON file
        json_list = []
        with open(file_path) as f:
            for line in f:
                # convert string to dictionary
                data = json.loads(line)
                # print the dictionary
                json_list.append(data)
        return json_list

    def open_csv_file(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for d in json_list:
                writer.writerow(d)

        # Review files
        # json_list = open_json_file(review_json)
        # keys = set().union(*(d.keys() for d in json_list))
        #
        # json_list = open_csv_file(review_csv)
        #
        # # business file
        # json_list = open_json_file(business_json)
        # keys = set().union(*(d.keys() for d in json_list))
        #
        # json_list = open_csv_file(business_csv)

    def business_eda(self):
        # review_csv = os.path.join(my_path, "../data/review_PA.csv")
        business_json = os.path.join(self.my_path, "../data/business.json")

        business = pd.read_json(business_json, lines=True)
        #Drop the instances where categories is null
        business = business.dropna(subset=["categories"])
        #Explode the categories column to check for count for each category
        business_cat = business.assign(categories=business.categories.str.split(', ')).explode('categories')
        cat_counts = business_cat.categories.value_counts()
        cat_counts = cat_counts.to_frame().reset_index()
        cat_counts.columns = ['Categories', 'Total_Count']
        print("Explode the categories column to check for count for each category", cat_counts)
        #Check for duplicate values
        print("business data duplicate", business.business_id.describe())

        #Visualize top 20 categories
        # Sort the DataFrame by count in descending order and select the top 20 categories
        top_categories = cat_counts.sort_values(by='Total_Count', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(18, 8))
        # Create a bar plot from the top_categories DataFrame
        ax = top_categories.plot.bar(x='Categories', y='Total_Count', rot=0, ax=ax)
        # Set the axis labels and title
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Top 20 Categories')
        # Rotate the x-axis labels by 90 degrees
        ax.tick_params(axis='x', rotation=45)
        # Show the plot
        plt.show()

        #Filter and keep the instances including 'Restaurant', 'Food', 'Sandwiches', 'Pizza', 'Fast Food', 'Breakfast & Brunch' catagories
        # Define the list of categories to filter
        categories = ['Restaurant', 'Food', 'Sandwiches', 'Pizza', 'Fast Food', 'Breakfast & Brunch']

        # Filter the rows that contain the desired categories
        print("Filter the rows that contain the desired categories")
        business = business[business['categories'].str.contains('|'.join(categories))]
        print(business.head(5))

        # Plot Restaurant statistics (Open versus Closed Restaurants)
        print("Plot Restaurant statistics (Open versus Closed Restaurants)")
        fig, ax = plt.subplots(figsize=(18, 8))
        # total number of restaurants vs open restaurants by state
        open_by_state = business.groupby('state')['is_open'].agg(['count', 'sum']).sort_values(by=['count'],
                                                                                               ascending=False)
        open_by_state.columns = ['Total_number_of_restaurants', 'Number_of_open_restaurants']
        ax = open_by_state.plot(kind='bar', rot=0, ax=ax)
        ax.set_xlabel('State')
        ax.set_ylabel('# Restaurants')
        ax.set_title('Open vs. total number of restaurants by state')
        plt.show()
        #Consider Businesses that are open
        business = business[business['is_open'] == 1]
        #Filter and consider Restaurants present in USA
        # 'business' dataframe is first reduced to US business only
        # drop the instances where zip-code is not a 5 digit zipcode

        zipcode_length = business.postal_code.astype(str).apply(
            len)  # drop entries where postal_code is not 5-digit from business
        business = business[zipcode_length == 5]

        # Keep only state that are present in USA.

        states = ['AL', 'AK', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'FM', 'GA', 'GU', 'HI', 'IA', 'ID',
                  'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MH', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                  'NH', 'NJ', 'NM', 'NV', 'NY', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'PW', 'RI', 'SC', 'SD', 'TN', 'TX',
                  'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']
        business = business[business.state.isin(states)]

        m = folium.Map(location=[39.8283, -98.5795], zoom_start=3.5)

        i = 0
        for idx, row in business.iterrows():
            if i > 1000: break
            folium.Marker(location=[row.latitude, row.longitude]).add_to(m)

            i += 1
        m.save('map.html')

        #Distribution of state in the dataset
        state_business_count = business.state.value_counts()
        state_business_count = state_business_count.to_frame().reset_index()
        state_business_count.columns = ['State', 'Count of restaurants']
        print(state_business_count)

        # the state of Pennsylvania showed highest count of restaurants.
        fig, ax = plt.subplots(figsize=(20, 10))

        # Define a list of Tableau colors
        tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F',
                          '#BAB0AC', '#7B4F8D', '#FFB347', '#4D4D4D', '#AEC7E8', '#FDBF6F']

        ax = state_business_count.plot.bar(x='State', y='Count of restaurants', rot=0, ax=ax, color=tableau_colors)
        # Set the axis labels and title
        ax.set_xlabel('States')
        ax.set_ylabel('Total_Count')
        ax.set_title('Distribution of State')

        # Rotate the x-axis labels by 90 degrees
        ax.tick_params(axis='x', rotation=45)
        # Show the plot
        plt.show()
        #The state of Pennsylvania has the highest number of resturants for Yelp dataset. Hence, we will be building a restaurant recommendation system for the state of Pennsylvania
        #Keeping data only for the state of Pennsylvania
        business_PA = business[business.state == 'PA']
        business_PA = business_PA.reset_index(drop=True)
        # We have performed it and create csv file to use.
        # Pennsyvania_restaurant_eda(business_PA)

    def Pennsyvania_restaurant_eda(business_PA):
        # Compute the mean review_count by name
        pa_restaurants_mean = business_PA.groupby('name').sum()['review_count'].reset_index()
        # Sort by descending mean review_count
        pa_restaurants_mean_sorted = pa_restaurants_mean.sort_values(by='review_count', ascending=False)
        # Keep only the top five restaurants
        pa_top_restaurants = pa_restaurants_mean_sorted.head(5)
        tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F',
                          '#BAB0AC']
        # Create the plot
        plt.figure()
        plt.barh(pa_top_restaurants['name'], pa_top_restaurants['review_count'], color=tableau_colors)
        plt.xlabel('Total Review Count')
        plt.title('Top 5 Restaurants in Pennsylvania by total Review Count')
        fig = plt.gcf()  # get the current figure
        fig.set_size_inches(12, 8)
        # Remove the legend
        plt.legend([])
        # Show the plot
        plt.show()

        #Restaurants with 5 star rating
        plt.figure(figsize=(20, 5))

        # restaurant distribution by rating

        business_PA.stars.plot(kind='hist', bins=9, range=(0.8, 5.2), rwidth=0.8, color='blue')
        plt.xlabel('Rating of the restaurant')
        plt.ylabel('Count of restaurants')
        plt.title('Restaurant distribution by rating')

        # Divide the contents in the attribute column into different columns as it contains features that may be important for building recommendation systems.
        business_PA.attributes.iloc[20]

        # define a function to extract the keys from the json objects
        def get_json_keys(json_string):
            keys = []
            json_data = json_string
            try:
                for key in json_data.keys():
                    keys.append(key)
                # print(keys)
                return keys
            except AttributeError:
                return keys

        # create an empty dictionary to store the counts for each key
        key_counts = {}

        # iterate over the rows in the dataframe
        for index, row in business_PA.iterrows():
            # get the json string from the attributes column
            attributes_json = row['attributes']
            # extract the keys from the json object
            keys = get_json_keys(attributes_json)
            # increment the count for each key
            for key in keys:
                if key in key_counts:
                    key_counts[key] += 1
                else:
                    key_counts[key] = 1

        # sort the keys by count in descending order
        sorted_keys = sorted(key_counts, key=key_counts.get, reverse=True)

        # print the top 20 keys by count
        top_keys = sorted_keys[:20]
        print(top_keys)
        # iterate over the rows again to populate the new columns
        for index, row in business_PA.iterrows():
            # get the json string from the attributes column
            attributes_json = row['attributes']
            # extract the keys and values from the json object
            json_data = attributes_json
            if index % 99 == 0: print(index)
            for key in top_keys:
                try:
                    if key in json_data:
                        business_PA.loc[index, key] = json_data[key]
                except TypeError:
                    continue
        #Checking Null values for these attributes
        null_values = business_PA.isna().sum()
        null_values_sorted = null_values.sort_values(ascending=True)
        print(null_values_sorted)
        #Divide the contents in the categories column into different columns as it may contain information relevant for building recommendation system
        subcategories = []
        for i in business_PA.index:
            elements = business_PA.categories[i].split(',')
            for element in elements:
                subcategories.append(element)
        subcategories = [x.strip(' ') for x in subcategories]
        sub_names = Counter(subcategories).most_common(20)
        sub_list = [x[0] for x in sub_names]
        print("categories column into different columns", sub_list)
        #Create the columns to store these sub-categories into the dataset
        print('Create the columns to store these sub-categories into the dataset')
        for col in sub_list:
            business_PA[col] = np.nan

        for i in sub_list:
            for index in business_PA.index:

                if i in business_PA.categories[index]:
                    business_PA[i][index] = 1
                else:
                    business_PA[i][index] = 0
       # Write businessPA csv file
       #  business_PA.to_csv(business_PA)

    def review_eda(self):
        # Read the CSV data into a DataFrame
        #Extracting CSV files from data folder
        review_csv = os.path.join(self.my_path, "../data/review_PA.csv")
        business_csv = os.path.join(self.my_path, "../data/business_PA.csv")

        df = pd.read_csv(review_csv)
        # Display the DataFrame
        print(df)
        print(df.head(3))
        df_bus = pd.read_csv(business_csv)

        #Remove the rows for Review CSV with NA values
        print("Before dropping NA", df.isna().sum())
        df = df.dropna()
        # Number of rows
        df.info()
        print("After Removing NA", df.isna().sum())
        #Top 5 businesses with highest count of 5 Star rating
        # Group the data by business_id and stars, and count the number of 5-star ratings
        grouped = df[df['stars'] == 5].groupby(['business_id'])['stars'].count().reset_index(name='count')
        # Sort the data by count, in descending order, and then by stars, in descending order
        sorted_df = grouped.sort_values(by=['count'], ascending=[False])
        # Get the top 5 businesses with the highest count of 5-star ratings
        top_businesses = sorted_df.head(5)['business_id'].tolist()
        business_categories = df_bus[df_bus['business_id'].isin(top_businesses)]['categories'].tolist()
        for i in range(len(business_categories)):
            business_categories[i] = ','.join(business_categories[i].split(',')[:5])
        # Print the result
        print("Top 5 businesses with the highest number of 5-star ratings:")
        print(top_businesses)
        print("Top 5 businesses Category with the highest number of 5-star ratings:")
        print(business_categories)

        # Filter the data to get only the rows with the top 5 businesses
        filtered_df = df[df['business_id'].isin(top_businesses)]

        # Group the data by business_id and stars, and count the number of 5-star ratings
        grouped_data = filtered_df[filtered_df['stars'] == 5].groupby(['business_id'])['stars'].count().reset_index(
            name='count')
        # Sort the data by count in descending order
        grouped_data = grouped_data.sort_values(by='count', ascending=False)
        # Plot the bar graph
        fig = px.bar(grouped_data, x='business_id', y='count', color=business_categories,
                     title='Number of 5-Star Ratings for Top 5 Businesses')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        fig.show()
        # Group the data by user_id and sum the number of useful votes
        grouped = df.groupby('user_id')['useful'].sum().reset_index(name='total_useful')

        # Sort the data by total_useful, in descending order
        sorted_df = grouped.sort_values(by='total_useful', ascending=False)

        # Get the top 5 most influential users
        top_users = sorted_df.head(5)

        # Create a bar graph for top_user vs the number of useful vote they have got using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_users['user_id'],
            y=top_users['total_useful'],
            marker_color='purple'
        ))
        fig.update_layout(
            title="Top 5 Most Influential Users Based on Total Number of Useful Votes Received",
            xaxis_title="User ID",
            yaxis_title="Total Number of Useful Votes Received",
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="#7f7f7f"
            )
        )
        #Impact of top 5 most influential on the businesses
        top_users = top_users['user_id'].tolist()
        # Filter the data to include only the reviews by the top 5 most influential users
        top_reviews = df[df['user_id'].isin(top_users)]
        # Group the data by business_id and calculate the average star rating
        grouped = top_reviews.groupby('business_id')['stars'].mean().reset_index(name='avg_stars')
        # Sort the data by avg_stars, in descending order
        sorted_df = grouped.sort_values(by='avg_stars', ascending=False)
        top_businesses = sorted_df.head(5)['business_id'].tolist()
        top_business_categories = df_bus[df_bus['business_id'].isin(top_businesses)]['categories'].tolist()
        for i in range(len(top_business_categories)):
            top_business_categories[i] = ','.join(top_business_categories[i].split(',')[:5])
        # Create a bar chart showing the average star rating of the top businesses reviewed by the top 5 most influential users
        fig = px.bar(sorted_df.head(5), x='business_id', y='avg_stars', color=top_business_categories,
                     title='Average Star Rating of Top Businesses Reviewed by Top 5 Most Influential Users')
        fig.show()
        # Yearly Trends in the Number of Reviews Received by Businesses
        # Convert the date column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Group the data by month and count the number of reviews
        yearly_reviews = df.groupby(pd.Grouper(key='date', freq='M'))['review_id'].count()

        # Plot the monthly trends in the number of reviews
        plt.plot(yearly_reviews.index, yearly_reviews.values)
        plt.xlabel('Yearly')
        plt.ylabel('Number of Reviews')
        plt.title('Yearly Trends in the Number of Reviews Received by Businesses')
        plt.show()

    # We have performed this function and filtered the data and save it in csv file
    #We don't need to run it again
    def review_preprocessing(self):
        review_csv = os.path.join(self.my_path, "../data/review_PA.csv")
        df_review = pd.read_csv(review_csv)
        print(df_review.isna().sum())
        df_review.drop(columns=['Unnamed: 0'])

        review_index = []
        for i in range(df_review.shape[0]):
            if df_review.iloc[i].business_id in unique_business_id:
                review_index.append(i)
        df_review = df_review.iloc[review_index, :]
        df_review.index = range(df_review.shape[0])
        year_ = [eval(x.split('-')[0]) for x in df_review.date]
        df_review['year'] = year_
        # WRITE THE FILE INTO CSV FORMAT
        df_review.to_csv('review_PA.csv')
        #Remove the rows with NA values in Review Dataframe
        #Get the Review Dataframe for the businesses in PA
        df = df_review[df_review['business_id'].isin(unique_business_ids)]
        unique_business_ids = get_unique_businessid()
        # Top 5 businesses with highest count of 5 Star rating
        # Group the data by business_id and stars, and count the number of 5-star ratings
        grouped = df[df['stars'] == 5].groupby(['business_id'])['stars'].count().reset_index(name='count')
        # Sort the data by count, in descending order, and then by stars, in descending order
        sorted_df = grouped.sort_values(by=['count'], ascending=[False])
        # Get the top 5 businesses with the highest count of 5-star ratings
        top_businesses = sorted_df.head(5)['business_id'].tolist()
        business_categories = df_bus[df_bus['business_id'].isin(top_businesses)]['categories'].tolist()
        for i in range(len(business_categories)):
            business_categories[i] = ','.join(business_categories[i].split(',')[:5])
        # Print the result
        print("Top 5 businesses with the highest number of 5-star ratings:")
        print(top_businesses)
        print("Top 5 businesses Category with the highest number of 5-star ratings:")
        print(business_categories)
    def get_unique_businessid(self):
        business_csv = os.path.join(self.my_path, "../data/business_PA.csv")
        df_bus = pd.read_csv(business_csv)
        # Filter the data for postal code PA
        df_pa = df_bus[df_bus['state'] == 'PA']
        # Get all the unique business_ids in the filtered data
        unique_business_ids = df_pa['business_id'].unique()
        return unique_business_ids

path = os.path.abspath(os.path.dirname(__file__))
preprocess = DataConverter(path)
preprocess.business_eda()
preprocess.review_eda()
