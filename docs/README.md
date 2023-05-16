# Restaurant Recommendation Project

## Content Based Filtering

## Collaborative Filtering

## Latent Matrix Factorisation


## Hybrid Model (Knowledge Based Filtering+ Content Filtering using LDA + User-Item based Collaborative Filtering using SVD) 

#### What is Hybrid Approach?

A hybrid approach recommendation system combines multiple recommendation techniques to provide more accurate and personalized recommendations. It leverages the strengths of different recommendation methods to overcome their limitations. For example, it can combine content-based filtering, collaborative filtering, and other techniques to generate recommendations based on user preferences, item attributes, and user-item interactions. By integrating various approaches, a hybrid recommendation system can enhance recommendation quality, improve coverage, and address the cold-start problem.

#### Knowledge Based Filtering:
In this, the user provided inputs as per thier preference such as location, cuisine etc.

#### Content-Based Filtering using LDA and Sentiment Analysis for Restaurants:
Content-based filtering is a recommendation technique that utilizes item features or attributes to make recommendations. In the context of restaurant recommendations, content-based filtering used user reviews and review count. By employing techniques like Latent Dirichlet Allocation (LDA) and sentiment analysis, the system  extracted meaningful topics from restaurant descriptions and analyze sentiment from user reviews. These extracted topics and sentiment were used to understand the content of restaurants and capture user preferences. Additionally, by incorporating a weighted score based on review count, the system can prioritize popular 50 restaurants and provide more reliable recommendations.

### User-Item Based Collaborative Filtering using SVD:
User-item based collaborative filtering is a popular recommendation technique that relies on user-item interactions to make recommendations. It builds a user-item matrix representing user preferences or ratings for items. Singular Value Decomposition (SVD) is a matrix factorization technique commonly used in collaborative filtering. It decomposes the user-item matrix into lower-dimensional matrices capturing latent factors. By applying SVD, the system can identify latent patterns and relationships between users and items. The learned latent factors can then be used to predict missing ratings or recommend items to users based on their similarities with other users or items. In this model, the data was filtered for top 50 restaurants recommeded by LDA model and then data was trained using SVD model.

#### Steps to Download the file:

 
