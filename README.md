# Amazon Product Recommender Systems
### Jin Hui Xu
### DATA606 Capstone Project

## Contents
- Presentation
  - PowerPoint presentation - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/05145b7b246b2f9bbba06f11ce1d5c85db1315ce/presentation/Project_Presentation_EDA_v1.1.pptx">PowerPoint Presentation Link</a>
  - Video presentation - <a href="https://youtu.be/sDxbj2RG-58">Youtube Presentation Link</a>
- [Introduction](#introduction)
- [Research Questions](#research-questions)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Review Data](#review-data)
  - [Product Data](#product-data)
  - [Merged Data](#merged-data)
  - Notebook - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/6b54bfbf6db39c300e6c0e9be89f1598b5abb49e/ipynb/DATA606_Part1.ipynb">EDA Notebook Link</a>
- [Methods](#methods)
  - [Base Model](#base-model) 
  - [Content-Based Filtering](#content-based-filtering)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Hybrid Model](#hybrid-model)
  - Notebooks 
    - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/15ba9741beb23100e4c726368bbe66ccf28297d0/ipynb/DATA606_Part2_KnowledgeBasedRecommender.ipynb">Base Model Notebook Link</a>
    - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/939699dc09932f34b6968b0dda38caa554c6bafe/ipynb/DATA606_Part2_ContentBasedFiltering.ipynb">Content-Based Filtering Notebook Link</a>
    - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/fed9ede8f383579e73fa2f00a5cd97448c168675/ipynb/DATA606_Part2_CollaborativeRecommender.ipynb">Collaborative Filtering Notebook Link</a>
    - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/1fefb0beb0018de01935edfdac798edd2736938e/ipynb/DATA606_Part3_HybridRecommender.ipynb">Hybrid Model Notebook Link</a>
- [System Integration/Deployment](#system-integrationdeployment)
  - Recommender Chatbot Website - <a href="https://data606project.pythonanywhere.com/" target="_blank">Link</a>
- [Outcomes](#outcomes)
- [References](#references)

## Introduction

As E-commerce becomes more and more popular in recent years, especially by the impact of the COVID-19 pandemic, many retailers and companies are switching their business models to adapt to the trend. In addition, with the rapid growth of big data technology, the cost of storage capacity to store enormous amounts of data from customers and visitors on the E-commerce sites decreases gradually. No matter the tech giant or start-up, all companies can make use of the gathered data to boost their business success to the next level.

In order to better understand the customers' behavior and enhance their shopping experience, product recommender systems can provide the ability to predict whether a specific customer would prefer a product by utilizing the enormous amounts of collected data and different machine learning approaches. There are several benefits that businesses can achieve using product recommender systems. It can drive traffic through personalized email messages to the store site and increase average order value. It also enhances the shopping experience by delivering relevant content based on personalized preferences. It can reduce workload for inventory management and boost work effectiveness. Moreover, it can create comprehensive reports to support making the right decision for business direction. Overall, product recommender systems not only boost the companies’ revenue but also increase customer satisfaction and loyalty.

The objective of this project is to analyze the reason for a product to be recommended and explore different data science methods and algorithms to implement product recommender systems. It will provide business owners or start-up companies a better idea of how recommender systems work and the related advantages.

## Research Questions
* What characteristics are useful to generate personalized recommendations?
* Which recommender systems algorithms/methods are most successful and practical?
* Can text data improve recommender systems' performance?

## Data

The data for this project is the Amazon Review Data (2018) which is collected by the University of California San Diego (https://nijianmo.github.io/amazon/index.html). The dataset includes reviews (rating, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features). It contains a total number of 233.1 million real reviews with the size of 34 gigabytes from Amazon. I would use a subset of this data due to the computing resource limitation. The smaller dataset is the subset of the data in a specific domain/category.

Data format:

Review data

![image](https://user-images.githubusercontent.com/24414472/156906115-ec2eaa39-9ca9-4541-8e30-2a634de36192.png)


Product Metadata

![image](https://user-images.githubusercontent.com/24414472/155896882-c7df8c3e-9dc4-4f1a-9a52-99497766c5ff.png)



## Exploratory Data Analysis

For this project, I choose the Appliance category from the entire Amazon review dataset because it has a moderate number of review and product records. This section summarize the EDA for the datasets. For a more comprehensive demonstration and visualization, please use the EDA notebook at <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/6b54bfbf6db39c300e6c0e9be89f1598b5abb49e/ipynb/DATA606_Part1.ipynb">EDA Notebook Link</a>.

### Review Data
There are a total of 602,777 review records in this category, and the dataset has 12 different features.

* The rating distribution graphs show that the overall ratings in this review data set are highly imbalanced, which contains more than 69% of 5 stars rating.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/review_rating2.png">

  Thus, in the following model development, we need to keep in mind that the accuracy metric may not be useful for evaluating the machine learning models; instead, precision, recall, and F1 score values could be suitable for model evaluation.

* There are a total of 515,650 distinct reviewers in this dataset, and the most active reviewer had reviewed 208 products with an average 4.98 rating score.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/reviewer_counts.png">

* The review year distribution graphs show that the reviews in this dataset are heavily collected after the year 2013, which can quite well represent the current generation customers' preferences. Besides, the review month distribution graphs show that the months are quite evenly distributed in the dataset, which we can conclude that the season doesn't play a significant role in the influence of the purchase of the appliances.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_year.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_month.png">

* Most of the reviews contain less than 100 words. The word counts distributions for each star rating review are similar, but if we look in to the detail of the box plot graph, we could see that negative or low star rating reviews have more texts entered. The box plot shows that the 5 stars rating reviews have the lowest interquartile range (IQR) compared to the other 4 ratings, which implies that it has average the shortest review text.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_word_distrubution.png" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/review_text_length_by_rating2.png"/> 


### Product Data

There are a total of 30,239 product records in this category, and the dataset has 19 different features.

* In the product dataset, the majority of the products (64.7%) are in the Tools & Home Improvement category, and the Appliances category also holds 21.5% in the dataset. In addition, there are a total of 2,762 brands, and Whirlpool is at the rank 1 position of amount of products.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/2d37430bab7fc3629760c8f44e9d256e1cfbc970/images/product_cat.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/375921a58ca59115db8a0c8e829cff119c1a74de/images/product_brand.png">

* Below is the list of the ranking of most reviewed products and their average ratings. Among 30,239 Appliances products, there are only 30,252 products were reviewed, so there are some products are not included in the product dataset. 

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_counts.png" />
  
* Within this list, the most reviewed product is General Electric MWF Refrigerator Water Filter, and the second most reviewed product is Samsung Genuine DA29-00020B Refrigerator Water Filter, 3 Pack. Both of them are Refrigerator Water Filters.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/1st_product.png" height='350'/> 
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/2nd_product.png" height='350'/> 

* The product text distribution histogram and box plot show that majority of the product text is less than 1000 words. There are only a few outliers that are greater than 2000 words,  so for future NLP model development, in order to reduce the padding size, we can choose a smaller number instead.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/9b4cb651e7e430486faa681c4e48af5d358d6fc6/images/product_text.png" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/product_text_boxplot2.png" /> 


* Besides, the word cloud shows that the most frequently used words for Appliances products are related to replacement, part, and model number.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/375921a58ca59115db8a0c8e829cff119c1a74de/images/product_wordcloud.png" /> 

### Merged Data

To find out more insights within this dataset, we can merge the cleaned review and product datasets.

* Most Reviewed Brands Distributions (top 10) graphs show that Whirlpool products have the rank 1 position of amount of reviews. However, there are some other brands in the list that are not in the list of top 10 product numbers, which means offering more products doesn't imply more sales and revenue.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/most_reviewed_brand2.png" /> 

* Besides, the below table shows the top 10 average rating brand (reviews > 5000) in the dataset. There are many brand only have a few reviews, and their average rating will definitely be higher than other brands with more reviews, so we only consider the brands with at least 5,000 reviews for this analysis.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/d1cad2dbfd4938c5db634e5c8ad83e92b0783edc/images/highest_rating_brand.png" /> 
  
  The result show that brand LintEater has the highest average rating 4.62 with over 6,000 reviews.
  Whereas Whirlpool has the rank 4 in this list, it also has the most review number and most products offered in the Appliance category. It means Whirlpool is doing great in   offering both overall product quality and quantity.

* This graph shows that LintEater has the best review per product ratio in the dataset. And most of the brands are not in the top ranking of the number of products, which again proves that offering more products doesn't imply more sales and revenue.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/Highest_ReviewProduct_Ratio_Brands.png" /> 

* The Most Reviewed Sub Category Distributions (top 10) graphs show that 37.9% of the reviews are in the Appliances Parts sub category, and the Accessories sub category also holds 17.1% in the dataset.

   <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/4274a5a201e957e307c31a467066e441ddbef00b/images/most_reviewed_subcategory2.png" /> 

## Methods

### Base Model 
<a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/15ba9741beb23100e4c726368bbe66ccf28297d0/ipynb/DATA606_Part2_KnowledgeBasedRecommender.ipynb">Base Model Notebook Link</a>

A base model is a simple knowledge-based recommender that takes user inputs such as product category, brand, release year, and targeted price to search for matching products. It usually doesn't leverage machine learning to provide recommendations. 

For this project, we are not deploying a model that takes user inputs like mentioned above. Instead, we sort the product lists by rating mean and review counts for a recommendation. This is the base model we would use if the users don't have a customer ID and product ID for our recommender system.

### Content-Based Filtering
<a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/939699dc09932f34b6968b0dda38caa554c6bafe/ipynb/DATA606_Part2_ContentBasedFiltering.ipynb">Content-Based Filtering Notebook Link</a>

The idea of content-based filtering is to find the similarity products based on either metadata or product description. The most feasible approach is to apply the cosine similarity method against the textual data to find the most similar products. I have applied this approach against both product description and metadata, and their recommendation results are very convincing. 

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/092e30589bd38cf5086afb1242981f5fbc016e2b/images/content_based_result.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/092e30589bd38cf5086afb1242981f5fbc016e2b/images/content_based_result2.png">

However, this approach is not a good fit for Web API because the matrix size is too large for local RAM or web hosting service storage and RAM, so we have to try using another method for Web API deployment for content-based filtering.

The topic modeling approach could be an alternative solution. Instead of computing the huge similarity matrix, leveraging a probabilistic topic model like LDA can cluster the entire product set into different topics or categories in our case. For creating recommendations, we can find the products that share the same topics among the product list and output the products with the highest probability scores. Although the output will be less precise than the cosine similarity models, it can be a good fit for Web API deployment.

To find out how many topics exist in our product dataset, coherence values analysis is applied and the output shows that topic number 9 has the best coherence score, so we will use k=9 for the final LDA model.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/092e30589bd38cf5086afb1242981f5fbc016e2b/images/LDA_coherence_scores.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/092e30589bd38cf5086afb1242981f5fbc016e2b/images/LDA_result.png">
  
The LDA model has a Coherence Score: 0.606, and The LDA topic modeling recommender did a fairly good job. However, the cosine similarity models are performing better and generating more concise recommendations. Since it is just adding extra topic number and probability columns to the dataset, the data file size would be much smaller than the cosine similarity model. Thus, we use the LDA model output for Web API depoyment.

### Collaborative Filtering
<a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/fed9ede8f383579e73fa2f00a5cd97448c168675/ipynb/DATA606_Part2_CollaborativeRecommender.ipynb">Collaborative Filtering Notebook Link</a>

Collaborative methods for recommender systems are methods based on past interactions recorded between users and items to generate new recommendations. The past user-item interactions represent the bases to detect similar users and/or similar items and to make predictions based on estimated proximities.

The class of collaborative filtering algorithms is divided into two sub-categories called memory-based and model-based approaches:
  - Memory-based approaches directly work with values of recorded interactions, assuming no model, and are essentially based on nearest neighbors search or KNN (i.e., find the closest users from a referenced user and suggest the most popular items among these neighbors)
  - Model-based approaches assume there is an underlying “generative” model that explains the user-item interactions and tries to identify it in order to make new predictions

For training the collaborative filtering model, we only consider the customer with at least 3 reviews in our dataset. This will increase the recommendation output accuracy. On the left-hand side, the graph shows the rating distribution for our selected training data. Since one of the objectives of this project is to find out if textual data can improve recommender systems' performance, I have also performed sentiment analysis against the review text and converted the sentiment polarity score into the same range as the review rating. The graph on the right shows that based on the sentiment of the review text, the rating distribution should not be that imbalanced. Becasue of the sentiment data is tend to be more normally distributed, I choose to use it for the machine learning models instead of the original rating.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/0838a59fc9dff663efb275ee5854c71190ab4435/images/compare.png">
  
Surprise is a good Python library to build collaborative recommendation system for both memory based and model based models. I have applied all the algorithms supported by Surprise and developed a deep neural network by TensorFlow and Keras to compare their performance results. The result table shows that SVDpp and SVD have a very close performance, but the training and testing time of SVDpp is longer than SVD. Thus, the SVD is our choice for the collaborative filtering recommendation model deployment.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/0838a59fc9dff663efb275ee5854c71190ab4435/images/collaborative_ml_result.png">

After the algorithm is picked, I have performed some parameter tuning using GridSearchCV to improve the performance as much as possible, and the final SVD model for deployment has a average 0.56 rmse testing score which is the best among all the collaborative filtering models. And the collaborative filtering model will take a reviewer ID as input to generate recommendation like this. Unlike content-based filtering, the results are personalized specifically for this customer.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/0838a59fc9dff663efb275ee5854c71190ab4435/images/final_SVD_rmse.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/0838a59fc9dff663efb275ee5854c71190ab4435/images/SVD_results.png">

### Hybrid Model
<a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/1fefb0beb0018de01935edfdac798edd2736938e/ipynb/DATA606_Part3_HybridRecommender.ipynb">Hybrid Model Notebook Link</a>

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/a6ee80eaec6256a12c862313fecd70ae936a65ef/images/filtering%20models.png">

The above two types of filtering have their own drawbacks such as the novelty problem of Content-based Filtering and the cold start problem of Collaborative Filtering, so in reality, more robust recommender systems like hybrid recommenders are often used. I have built a hybrid recommender that combines Content-based Filtering and Collaborative Filtering to overcome the drawbacks and improve overall performance.

The goal of our Hybrid Model is to generate recommendations based on both content-based filtering and collaborative filtering. It will take both reviewer ID and product ID as input, and first get 100 recommendation results from the content-based filtering model, then input the reviewer ID and the recommended product IDs from the Content-based filtering model to the Collaborative filtering model. This model will generate recommendations that meet product similarities and customer personality as much as possible.

Of course, if either ID is missing from the input, our system can handle it by calling its "Child Models" to generate recommendations respectively. If no input IDs are entered, then it will use our base model for the recommendation.

As you can see, the recommendation result shows that the hybrid model is suggesting more products that are similar to the product ID B0001YH10C for customer ID A1CY6CQC5HPQGL because it takes individual advantage of content-based and collaborative filtering.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/7e9906a5566c32ca33e7a64cd1bc0803a91b33a6/images/hybrid_result.png">

## System Integration/Deployment

For recommendation system deployment, a user interface was developed by integrating with multiple platforms and servers. This section will illustrate the system integration details and provide a live recommendation website at the end.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/c1cf47c7c051d36be2c7c63f0d5a7ab17689e830/images/system%20architecture%20diagram.drawio.png" />
  
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/c1cf47c7c051d36be2c7c63f0d5a7ab17689e830/images/chat%20flow.drawio.png" />
  
The recommendation chatbot website is hosted at: https://data606project.pythonanywhere.com/

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/a33a54b890393936b50ff188b1c77e4d45b3c37d/images/chatbot_site.png" /> 

## Outcomes
#### What outcomes do you intend to achieve (better understanding of problems, tools to help solve problems, predictive analytics with practicle applications, etc)?

I intend to achieve through this project is to develop product recommender systems/models that can accurately predict customers' preferences, identify the most useful characteristics to promote certain products to customers, understand the role of text data in recommender systems, provide a website and Chatbot to assist amazon users to make purchase decisions, and provide a comprehensive report of recommender systems for the business owners.

## References
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019 
http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf

Doshi, S. (2019, February 20). Brief on Recommender Systems. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/brief-on-recommender-systems-b86a1068a4dd 

Engineering@ZenOfAI. (2019, August 7). Creating chatbot with Webhooks using python (FLASK) and dialogflow. Medium. Retrieved March 5, 2022, from https://medium.com/zenofai/creating-chatbot-using-python-flask-d6947d8ef805

BANIK, R. O. U. N. A. K. (2018). Hands-on recommendation systems with Python: Start building powerful and personalized, ... recommendation engines with python. PACKT Publishing Limited. 


