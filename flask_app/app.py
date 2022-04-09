# /index.py
from flask import Flask, request, jsonify, render_template
import os
import dialogflow
import requests
import json
import pusher
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/product')
def product():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    product_df = pd.read_csv(data_url, compression='zip')
    display_data = product_df[['asin', 'ori_title', 'brand', 'main_cat']].sample(n = 1000)
    product_data = display_data.values
    return render_template('product.html', Data=product_data)


@app.route('/reviewer')
def reviewer():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "cr_sentiment_result.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    df = pd.DataFrame(product_df['reviewerID'].unique().tolist())
    df.columns=['reviewerID']

    display_data = df.sample(n=1000)
    product_data = display_data.values
    return render_template('reviewer.html', Data=product_data)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    query_result = req.get('queryResult')

    if query_result.get('action') == 'Content_based'or query_result.get('action') == 'recommender.recommender-no.recommender-no-yes.recommender-no-yes-custom':
        product_id = query_result.get('parameters').get('ProductID')
        product_id = product_id.replace(" ", "")
        rec = get_rec(product_id)
        if product_id != '':
            reply = {
                "fulfillmentText": rec
            }
            return jsonify(reply)
    elif query_result.get('action') == 'Collaborative' or query_result.get('action') == 'recommender.recommender-yes.recommender-yes-custom':
        product_id = query_result.get('parameters').get('ReviewerID')
        product_id = product_id.replace(" ", "")
        rec = get_rec_c(product_id)
        if product_id != '':
            reply = {
                "fulfillmentText": rec
            }
            return jsonify(reply)
    elif query_result.get('action') == 'product.detail':
        product_id = query_result.get('parameters').get('ProductID')
        product_id = product_id.replace(" ", "")
        detail = get_detail(product_id)
        if product_id != '':
            reply = {
                "fulfillmentText": detail
            }
            return jsonify(reply)

    reply = {
        "fulfillmentText": ''
    }

    return jsonify(reply)


def get_detail(id):
    id = id.lower()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    input_id = id
    if len(product_df[product_df['asin'].str.lower() == input_id]) == 0:
        detail = "Couldn't find a product for this ID, please try a different one."
    else:
        input_title = product_df[product_df['asin'].str.lower() == input_id]['ori_title'].item()
        link = '<a class="btn btn-success" href="{}" target="_blank">Find this product at Amazon</a>'.format(
            'https://www.amazon.com/dp/' + input_id.upper())

        if len(product_df[product_df['asin'].str.lower() == input_id]['imageURLHighRes'].tolist()[0]) == 0:
            detail = '{}, {}<br><br>{}'.format(input_id.upper(), input_title, link)
        else:
            input_img_url = str(product_df[product_df['asin'].str.lower() == input_id]['imageURLHighRes'].tolist()[0]).replace('[','').replace(']','').replace("'",'').split(",")[0]
            print(input_img_url)
            img = '<img src="{}" alt="Product Image not Available" class="img_size">'.format(input_img_url)
            detail = '{}, {}<br><br>{}<br><br>{}'.format(input_id.upper(), input_title, img, link)

    return detail


def get_rec_c(id):
    #url = 'https://github.com/JinHuiXu1991/Jin_DATA606/blob/main/cleaned_data/cleaned_amazon_product.zip?raw=true'
    #product_df = pd.read_csv(url, compression='zip')
    #product_df = product_df.fillna('')
    id = id.lower()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "cr_sentiment_result.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    input_id = id
    if len(product_df[product_df['reviewerID'].str.lower() == input_id]) == 0:
        rec = "Couldn't find this customer, please try a different customer ID."
    else:
        products = sentiment_collaborative_recommender(input_id, product_df)

        rec = 'Collaborative Filerting Recommender Result for customer {}: <br>'.format(input_id.upper())
        for i, product in enumerate(products):
            link = '<a class="" href="{}" target="_blank">{}</a>'.format(
                'https://www.amazon.com/dp/' + product[1], product[1])
            rec += '{}. {}, {}<br>'.format(i + 1, link, product[2])

    return rec


def sentiment_collaborative_recommender(id, df):
    id = id.lower()
    # return the top 10 most similar product
    return df[df['reviewerID'].str.lower() == id].values.tolist()


def get_rec(id):
    #url = 'https://github.com/JinHuiXu1991/Jin_DATA606/blob/main/cleaned_data/cleaned_amazon_product.zip?raw=true'
    #product_df = pd.read_csv(url, compression='zip')
    #product_df = product_df.fillna('')
    id = id.lower()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    input_id = id
    if len(product_df[product_df['asin'].str.lower() == input_id]) == 0:
        rec = "Couldn't find any similar product, please try a different product ID."
    else:
        input_title = product_df[product_df['asin'].str.lower() == input_id]['ori_title'].item()

        asin, title = topic_modeling_recommender(input_id, product_df)

        rec = 'Topic Modeling Recommender Result for {}, {}: <br>'.format(input_id.upper(), input_title)
        for i in range(0, 10):
            link = '<a class="" href="{}" target="_blank">{}</a>'.format(
                'https://www.amazon.com/dp/' + asin[i], asin[i])
            rec += '{}. {}, {}<br>'.format(i + 1, link, title[i])

    return rec


def topic_modeling_recommender(id, df):
    id = id.lower()

    # get the input product topic number
    topic_num = df[df['asin'].str.lower() == id]['topic_num'].item()

    # remove the input product from the recommendation data
    exclude_input_df = df.copy()
    exclude_input_df = exclude_input_df[exclude_input_df['asin'].str.lower() != id]

    # get the top 10 Probability product for the matching topic number
    output_df = exclude_input_df[exclude_input_df['topic_num'] == topic_num].sort_values('probability', ascending=False).head(10)

    # get the product indices
    product_indices = output_df.index.tolist()

    # return the top 10 most similar product
    return df['asin'].iloc[product_indices].tolist(), df['ori_title'].iloc[product_indices].tolist()


def detect_intent_texts(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    if text:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(
            session=session, query_input=query_input)
        return response.query_result.fulfillment_text


@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
    response_text = {"message":  fulfillment_text}
    return jsonify(response_text)


# run Flask app
if __name__ == "__main__":
    app.run()
