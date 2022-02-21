# Amazon Product Recommender Systems
### Jin Hui Xu
### DATA606 Capstone Project

## Introduction

As E-commerce becomes more and more popular in recent years, especially by the impact of the COVID-19 pandemic, many retailers and companies are switching their business models to adapt to the trend. In addition, with the rapid growth of big data technology, the cost of storage capacity to store enormous amounts of data from customers and visitors on the E-commerce sites decreases gradually. No matter the tech giant or start-up, all companies can make use of the gathered data to boost their business success to the next level.

In order to better understand the customers' behavior and enhance their shopping experience, product recommender systems can provide the ability to predict whether a specific customer would prefer a product by utilizing the enormous amounts of collected data and different machine learning approaches. There are several benefits that businesses can achieve using product recommender systems. It can drive traffic through personalized email messages to the store site and increase average order value. It also enhances the shopping experience by delivering relevant content based on personalized preferences. It can reduce workload for inventory management and boost work effectiveness. Moreover, it can create comprehensive reports to support making the right decision for business direction. Overall, product recommender systems not only boost the companies’ revenue but also increase customer satisfaction and loyalty.

The objective of this project is to analyze the reason for a product to be recommended and explore different data science methods and algorithms to implement product recommender systems. It will provide business owners or start-up companies a better idea of how recommender systems work and the related advantages.


## Data

The data for this project is the Amazon Review Data (2018) which is collected by the University of California San Diego (https://nijianmo.github.io/amazon/index.html). The dataset includes reviews (rating, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features). It contains a total number of 233.1 million real reviews with the size of 34 gigabytes from Amazon. I would use a subset of this data due to the computing resource limitation. The smaller dataset is the subset of the data in a specific domain/category.

## Exploratory Data Analysis

For this project, I choose the Appliance category from the entire Amazon review dataset because it has a moderate number of review and product records. This section summarize the EDA for the datasets, for a more comprehensive demonstration and visualization, please use the EDA notebook at https://github.com/JinHuiXu1991/Jin_DATA606/blob/dcbf6d71c8a12ea498c7943c09cacdee08705d3f/ipynb/DATA606_Part1.ipynb

### Review Data
There are a total of 602,777 review records in this category, and the dataset has 12 different features.

The rating distribution graphs show that the overall ratings in this review data set are highly imbalanced, which contains more than 69% of 5 stars rating.

<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/dcbf6d71c8a12ea498c7943c09cacdee08705d3f/images/review_rating.png">

Thus, in the following model development, we need to keep in mind that the accuracy metric may not be useful for evaluating the machine learning models; instead, precision, recall, and F1 score values could be suitable for model evaluation.

There are a total of 515,650 distinct reviewers in this dataset, and the most active reviewer had reviewed 208 products with an average 4.98 rating score.

<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/reviewer_counts.png">

The review year distribution graphs show that the reviews in this dataset are heavily collected after the year 2013, which can quite well represent the current generation customers' preferences. Besides, the review month distribution graphs show that the months are quite evenly distributed in the dataset, which we can conclude that the season doesn't play a significant role in the influence of the purchase of the appliances.

<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_year.png">
<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_month.png">

Most of the reviews contain less than 100 words. The word counts distributions for each star rating review are similar, but if we look in to the detail of the box plot graph, we could see that negative or low star rating reviews have more texts entered. The box plot shows that the 5 stars rating reviews have the lowest interquartile range (IQR) compared to the other 4 ratings, which implies that it has average the shortest review text.

<p float="left">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_word_distrubution.png" width="550" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_text_length_by_rating.png" width="450" /> 
</p>

### Product Data

There are a total of 30,445 product records in this category, and the dataset has 19 different features.

In the product dataset, the majority of the products (64.6%) are in the Tools & Home Improvement category, and the Appliances category also holds 21.5% in the dataset. In additiion, there are a total of 2,762 brands, and Whirlpool is at the rank 1 position of amount of products.

<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_cat.png">
<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_brand.png">

Below is the list of the ranking of most reviewed products and their average ratings. Among 30,445 Appliances products, there are only 30,252 products were reviewed. Within this list, the most reviewed product is General Electric MWF Refrigerator Water Filter, and the second most reviewed product is Samsung Genuine DA29-00020B Refrigerator Water Filter, 3 Pack. Both of them are Refrigerator Water Filters.

<p float="left">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_counts.png" width="250" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/1st_product.png" width="320" /> 
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/2nd_product.png" width="420" /> 
</p>


## References
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019 
http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf

Doshi, S. (2019, February 20). Brief on Recommender Systems. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/brief-on-recommender-systems-b86a1068a4dd 
