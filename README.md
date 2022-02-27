# Amazon Product Recommender Systems
### Jin Hui Xu
### DATA606 Capstone Project

### PowerPoint presentation - <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/92a8591a599936073c53b06474798432e548ea63/presentation/Project_Presentation_EDA_v1.1.pptx">PowerPoint Presentation Link</a>

## Contents
- [Introduction](#introduction)
- [Research Questions](#research-questions)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Review Data](#review-data)
  - [Product Data](#product-data)
  - [Merged Data](#merged-data)
  - Notebook: <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/2def288e220758e429219f0b2adde8b4a051fb07/ipynb/DATA606_Part1.ipynb">EDA Notebook Link</a>
- [Methods](#methods)
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

![image](https://user-images.githubusercontent.com/24414472/155896401-fcd6c4d8-ce08-4a9c-b431-71b6d9a15331.png)


Product Metadata

![image](https://user-images.githubusercontent.com/24414472/155896882-c7df8c3e-9dc4-4f1a-9a52-99497766c5ff.png)



## Exploratory Data Analysis

For this project, I choose the Appliance category from the entire Amazon review dataset because it has a moderate number of review and product records. This section summarize the EDA for the datasets. For a more comprehensive demonstration and visualization, please use the EDA notebook at <a href="https://github.com/JinHuiXu1991/Jin_DATA606/blob/2def288e220758e429219f0b2adde8b4a051fb07/ipynb/DATA606_Part1.ipynb">EDA Notebook Link</a>.

### Review Data
There are a total of 602,777 review records in this category, and the dataset has 12 different features.

* The rating distribution graphs show that the overall ratings in this review data set are highly imbalanced, which contains more than 69% of 5 stars rating.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/a1a2261bc6bf8889eaa5f258c3c57e2c9d056c5e/images/review_rating.png">

  Thus, in the following model development, we need to keep in mind that the accuracy metric may not be useful for evaluating the machine learning models; instead, precision, recall, and F1 score values could be suitable for model evaluation.

* There are a total of 515,650 distinct reviewers in this dataset, and the most active reviewer had reviewed 208 products with an average 4.98 rating score.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/reviewer_counts.png">

* The review year distribution graphs show that the reviews in this dataset are heavily collected after the year 2013, which can quite well represent the current generation customers' preferences. Besides, the review month distribution graphs show that the months are quite evenly distributed in the dataset, which we can conclude that the season doesn't play a significant role in the influence of the purchase of the appliances.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_year.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_month.png">

* Most of the reviews contain less than 100 words. The word counts distributions for each star rating review are similar, but if we look in to the detail of the box plot graph, we could see that negative or low star rating reviews have more texts entered. The box plot shows that the 5 stars rating reviews have the lowest interquartile range (IQR) compared to the other 4 ratings, which implies that it has average the shortest review text.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_word_distrubution.png" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/ee921b2d64cc56a03b8bdb0fec45b46f1f6346e8/images/review_text_length_by_rating.png"/> 


### Product Data

There are a total of 30,445 product records in this category, and the dataset has 19 different features.

* In the product dataset, the majority of the products (64.6%) are in the Tools & Home Improvement category, and the Appliances category also holds 21.5% in the dataset. In addition, there are a total of 2,762 brands, and Whirlpool is at the rank 1 position of amount of products.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_cat.png">
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_brand.png">

* Below is the list of the ranking of most reviewed products and their average ratings. Among 30,445 Appliances products, there are only 30,252 products were reviewed. 

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/product_counts.png" />
  
* Within this list, the most reviewed product is General Electric MWF Refrigerator Water Filter, and the second most reviewed product is Samsung Genuine DA29-00020B Refrigerator Water Filter, 3 Pack. Both of them are Refrigerator Water Filters.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/1st_product.png" height='350'/> 
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/07e65b0c76686b1e612ef3aa0f26c56e47026c69/images/2nd_product.png" height='350'/> 

* The product text distribution histogram and box plot show that majority of the product text is less than 1000 words. There are only a few outliers that are greater than 2000 words,  so for future NLP model development, in order to reduce the padding size, we can choose a smaller number instead.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/9b4cb651e7e430486faa681c4e48af5d358d6fc6/images/product_text.png" />
  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/9b4cb651e7e430486faa681c4e48af5d358d6fc6/images/product_text_boxplot.png" /> 


* Besides, the word cloud shows that the most frequently used words for Appliances products are related to replacement, part, and model number.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/9b4cb651e7e430486faa681c4e48af5d358d6fc6/images/product_wordcloud.png" /> 

### Merged Data

To find out more insights within this dataset, we can merge the cleaned review and product datasets.

* Most Reviewed Brands Distributions (top 10) graphs show that Whirlpool products have the rank 1 position of amount of reviews. However, there are some other brands in the list that are not in the list of top 10 product numbers, which means offering more products doesn't imply more purchasing and revenue.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/d1cad2dbfd4938c5db634e5c8ad83e92b0783edc/images/most_reviewed_brand.png" /> 

* Besides, the below table shows the top 10 average rating brand (reviews > 5000) in the dataset. There are many brand only have a few reviews, and their average rating will definitely be higher than other brands with more reviews, so we only consider the brands with at least 5,000 reviews for this analysis.

  <img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/d1cad2dbfd4938c5db634e5c8ad83e92b0783edc/images/highest_rating_brand.png" /> 
  
  The result show that brand LintEater has the highest average rating 4.62 with over 6,000 reviews.
  Whereas Whirlpool has the rank 4 in this list, it also has the most review number and most products offered in the Appliance category. It means Whirlpool is doing great in   offering both overall product quality and quantity.

## Methods
#### What variables/measures do you plan to use in your analysis (variables should be tied to the questions in #3)?
<img src="https://github.com/JinHuiXu1991/Jin_DATA606/blob/a6ee80eaec6256a12c862313fecd70ae936a65ef/images/filtering%20models.png">
I plan to use both Content-based Filtering and Collaborative Filtering for the product recommender systems in this project. For Content-based Filtering, the variables should be the product metadata like feature, description, price, brand, and categories. For Collaborative Filtering, more variables from the review data should be used, such as overall rating, reviewText, and summary.

#### What kinds of techniques/models do you plan to use (for example, clustering, NLP, ARIMA, etc.)? How do you plan to develop/apply ML and how you evaluate/compare the performance of the models?

I plan to use the Cosine similarity model, Matrix Factorization, KNN. Besides, NLP models like TF-IDF, Naive Bayes, LSTM could be used against the text data. For evaluate/compare the performance of the models, I plan to apply Root Mean Squared Error (RMSE) and Decision support metrics (Precision, Recall, F1).

For Content-based Filtering, I will apply the cosine similarity method against the product metadata to identify the similar products for the given one. The main feature that will be used for this model is from the product metadata like description, price, salesRank, brand, categories, and product features. Since some of them are textual data, NLP techniques like tokenization and TF-IDF vectorization will be applied. 

For Collaborative Filtering, I will apply the matrix factorization method against the review data. The main feature that will be used for this approach is from the review data like user id, product id, and the rating score. To perform matrix analysis, the cosine similarity method could be applied again, and several machine learning algorithms will be used such as KNN and Singular value decomposition (SVD). KNN can group users into a cluster and only consider the same cluster user for product recommendation. SVD can break down a matrix into the product of a few smaller matrices to reveal the user connections and to discover relationships between items. Moreover, deep learning techniques could also be applied for Collborative Filtering, Neural Network method can take the user-item matrix or review textual data for predicting a score for recommending.

The above two types of filtering have their own drawbacks such as the novelty problem of Content-based Filtering and the cold start problem of Collaborative Filtering, so in reality, more robust recommender systems like hybrid recommenders are often used. I plan to build a hybrid recommender that combines Content-based Filtering and Collaborative Filtering to overcome the drawbacks and improve overall performance.

## Outcomes
#### What outcomes do you intend to achieve (better understanding of problems, tools to help solve problems, predictive analytics with practicle applications, etc)?

I intend to achieve through this project is to develop product recommender systems/models that can accurately predict customers' preferences, identify the most useful characteristics to promote certain products to customers, understand the role of text data in recommender systems, and provide a comprehensive report of recommender systems for the business owners.

## References
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019 
http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf

Doshi, S. (2019, February 20). Brief on Recommender Systems. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/brief-on-recommender-systems-b86a1068a4dd 
