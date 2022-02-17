# Amazon Product Recommender Systems
### Jin Hui Xu
### DATA606 Capstone Project

## Introduction
#### What is your issue of interest (provide sufficient background information)?  Why is this issue important to you and/or to others?

As E-commerce becomes more and more popular in recent years, especially by the impact of the COVID-19 pandemic, many retailers and companies are switching their business models to adapt to the trend. In addition, with the rapid growth of big data technology, the cost of storage capacity to store enormous amounts of data from customers and visitors on the E-commerce sites decreases gradually. No matter the tech giant or start-up, all companies can make use of the gathered data to boost their business success to the next level.

In order to better understand the customers' behavior and enhance their shopping experience, product recommender systems can provide the ability to predict whether a specific customer would prefer a product by utilizing the enormous amounts of collected data and different machine learning approaches. There are several benefits that businesses can achieve using product recommender systems. It can drive traffic through personalized email messages to the store site and increase average order value. It also enhances the shopping experience by delivering relevant content based on personalized preferences. It can reduce workload for inventory management and boost work effectiveness. Moreover, it can create comprehensive reports to support making the right decision for business direction. Overall, product recommender systems not only boost the companies’ revenue but also increase customer satisfaction and loyalty.

The objective of this project is to analyze the reason for a product to be recommended and explore different data science methods and algorithms to implement product recommender systems. It will provide business owners or start-up companies a better idea of how recommender systems work and the related advantages.

#### What questions do you have in mind and would like to answer?
* What characteristics are useful to generate personalized recommendations? 
* Which recommender systems algorithms/methods are most successful and practical?
* Can text data improve recommender systems' performance? 

## Data
#### Where do you get the data to analyze and help answer your questions (creditability of source, quality of data, size of data, attributes of data. etc.)?

The data for this project is the Amazon Review Data (2018) which is collected by the University of California San Diego (https://nijianmo.github.io/amazon/index.html). The dataset includes reviews (rating, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features). It contains a total number of 233.1 million real reviews with the size of 34 gigabytes from Amazon. I would use a subset of this data due to the computing resource limitation. The smaller dataset is the subset of the data in which all users and items have at least 5 reviews. 

The smaller dataset might result in a lack of sufficient data for building a Collaborative Filtering model because it requires a large amount of purchase history for the same product to generate recommendations. In that case, I might use the full review dataset but with the rating feature only to resolve the computing power issue.

Data format:

Review data
* reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
* asin - ID of the product, e.g. 0000013714
* reviewerName - name of the reviewer
* vote - helpful votes of the review
* style - a disctionary of the product metadata, e.g., "Format" is "Hardcover"
* reviewText - text of the review
* overall - rating of the product
* summary - summary of the review
* unixReviewTime - time of the review (unix time)
* reviewTime - time of the review (raw)
* image - images that users post after they have received the product

Product Metadata
* asin - ID of the product, e.g. 0000031852
* title - name of the product
* feature - bullet-point format features of the product
* description - description of the product
* price - price in US dollars (at time of crawl)
* imageURL - url of the product image
* imageURL - url of the high resolution product image
* related - related products (also bought, also viewed, bought together, buy after viewing)
* salesRank - sales rank information
* brand - brand name
* categories - list of categories the product belongs to
* tech1 - the first technical detail table of the product
* tech2 - the second technical detail table of the product
* similar - similar product table

#### What will be your unit of analysis (for example, patient, organization, or country)? Roughly how many units (observations) do you expect to analyze?

The unit of analysis for this project is the users and the products of the Amazon website. I would expect to analyze an entire products base for a chosen category, for example, Video Games, a total number of 84893 products.

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
