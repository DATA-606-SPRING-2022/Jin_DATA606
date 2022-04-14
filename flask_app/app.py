# /index.py
from flask import Flask, request, jsonify, render_template
import os
import dialogflow
import requests
import json
import pusher
import pandas as pd
from surprise import dump
from surprise import Dataset
from surprise import Reader
from collections import defaultdict

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
    data_url = os.path.join(SITE_ROOT, "static/data", "final_cr_sentiment_data.zip")
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

    if query_result.get('action') == 'Content_based' or query_result.get('action') == 'recommender.recommender-no.recommender-no-yes.recommender-no-yes-custom':
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
    data_url = os.path.join(SITE_ROOT, "static/data", "final_cr_sentiment_data.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    input_id = id
    if len(product_df[product_df['reviewerID'].str.lower() == input_id]) == 0:
        rec = "Couldn't find this customer, please try a different customer ID."
    else:
        rec = hybrid_recommender(reviewerID=input_id.upper(), productID='')
        '''
        asin, title = sentiment_collaborative_recommender(input_id, product_df)

        rec = 'Collaborative Filerting Recommender Result for customer {}: <br>'.format(input_id.upper())
        for i in range(0, 10):
            link = '<a class="" href="{}" target="_blank">{}</a>'.format(
                'https://www.amazon.com/dp/' + asin[i], asin[i])
            rec += '{}. {}, {}<br>'.format(i + 1, link, title[i])
        '''
    return rec


def load_model(model_filename):
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    return loaded_model


def get_rec_user(uid, input_df):
    input_id = uid
    data1 = [input_id]
    data2 = input_df['asin'].unique().tolist()

    df = pd.DataFrame(data1)
    df.columns =['reviewerID']

    df1 = pd.DataFrame(data2)
    df1.columns =['asin']
    # filter out reviewed products
    reviewed_product = input_df[input_df['reviewerID'] == input_id].asin.unique().tolist()
    df1 = df1[~df1['asin'].isin(reviewed_product)]

    # Now to perform cross join, we will create
    # a key column in both the DataFrames to
    # merge on that key.
    df['key'] = 1
    df1['key'] = 1

    # to obtain the cross join we will merge
    # on the key and drop it.
    result = pd.merge(df, df1, on ='key')

    result['overall']=0.0
    del result['key']
    return result


def collaborative_SVD_recommender(predictions, product_df, top_num=10, LDA_result=None):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
      if LDA_result is not None:
        if iid in LDA_result:
          top_n[uid].append((iid, est))
      else:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:top_num]

    for uid, user_ratings in top_n.items():
      result = [iid for (iid, _) in user_ratings]

    return result, product_df[product_df['asin'].isin(result)]['ori_title'].tolist()


def sentiment_collaborative_recommender(id, df):
    id = id.upper()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/model", "final_cr_sentiment_model.pickle")
    loaded_model = load_model(data_url)

    data_url2 = os.path.join(SITE_ROOT, "static/data", "final_cr_sentiment_data.zip")
    product_df = pd.read_csv(data_url2, compression='zip')

    data_url3 = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    title_df = pd.read_csv(data_url3, compression='zip')

    reader = Reader()
    result = get_rec_user(id, product_df)
    valid_Dataset = Dataset.load_from_df(result, reader)

    testset = valid_Dataset.df.values.tolist()
    predictions = loaded_model.test(testset)
    asin, title = collaborative_SVD_recommender(predictions, title_df)
    return asin, title
    # return the top 10 most similar product
    # return df[df['reviewerID'].str.lower() == id].values.tolist()


def hybrid_recommender(reviewerID="", productID=""):
    # load the final pretrained SVD model for collaborative filtering
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/model", "final_cr_sentiment_model.pickle")
    loaded_model = load_model(data_url)

    data_url2 = os.path.join(SITE_ROOT, "static/data", "final_cr_sentiment_data.zip")
    final_merged_df = pd.read_csv(data_url2, compression='zip')

    data_url3 = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    product_df = pd.read_csv(data_url3, compression='zip')

    reader = Reader()

    # both product ID and reviewer ID entered, do hybrid
    if productID != "" and reviewerID != "":
        rec = 'Recommending based on hybrid system: <br>'
        lda_asin, _ = topic_modeling_recommender(productID, product_df, 100)
        result = get_rec_user(reviewerID, final_merged_df)
        valid_Dataset = Dataset.load_from_df(result, reader)
        testset = valid_Dataset.df.values.tolist()
        predictions = loaded_model.test(testset)
        asin, title = collaborative_SVD_recommender(predictions, product_df, 10, lda_asin)

    # product ID entered and no reviewer ID entered, do content based
    elif productID != "" and reviewerID == "":
        input_title = product_df[product_df['asin'].str.lower() == productID.lower()]['ori_title'].item()
        rec = 'Recommending based on content-based filtering for product {}, {}: <br>'.format(productID.upper(), input_title)
        asin, title = topic_modeling_recommender(productID, product_df, 10)

    # no product ID entered and reviewer ID entered, do collaborative
    elif productID == "" and reviewerID != "":
        rec = 'Recommending based on collaborative filtering for customer {}: <br>'.format(reviewerID.upper())
        result = get_rec_user(reviewerID, final_merged_df)
        valid_Dataset = Dataset.load_from_df(result, reader)
        testset = valid_Dataset.df.values.tolist()
        predictions = loaded_model.test(testset)
        asin, title = collaborative_SVD_recommender(predictions, product_df, 10)

    # no product ID entered and no reviewer ID entered, do rating rank
    else:
        rec = 'Recommending based on product rating rank: <br>'
        data_url4 = os.path.join(SITE_ROOT, "static/data", "cleaned_amazon_review.zip")
        review_df = pd.read_csv(data_url4, compression='zip')
        merged_df = review_df.merge(product_df, on='asin', how='left')
        product_grouped = merged_df.groupby('asin').size().reset_index(name='counts')
        product_grouped_rating = merged_df.groupby('asin')['overall'].sum().reset_index(name='overall_sum')
        product_grouped['rating_mean'] = product_grouped_rating['overall_sum'] / product_grouped['counts']

        rating_df = product_df.merge(product_grouped, on='asin', how='left')
        output_df = rating_df.sort_values(['rating_mean', 'counts'], ascending=[False, False]).head(10)

        # get the product indices
        product_indices = output_df.index.tolist()

        # return the top 10 most similar product
        asin, title = product_df['asin'].iloc[product_indices].tolist(), product_df['ori_title'].iloc[
            product_indices].tolist()

    for i in range(0, 10):
        link = '<a class="" href="{}" target="_blank">{}</a>'.format(
            'https://www.amazon.com/dp/' + asin[i], asin[i])
        rec += '{}. {}, {}<br>'.format(i + 1, link, title[i])

    return rec


def get_rec(id):
    id = id.lower()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    data_url = os.path.join(SITE_ROOT, "static/data", "Content_based_LDA_output.zip")
    product_df = pd.read_csv(data_url, compression='zip')

    input_id = id
    if len(product_df[product_df['asin'].str.lower() == input_id]) == 0:
        rec = "Couldn't find any similar product, please try a different product ID."
    else:
        rec = hybrid_recommender(reviewerID='', productID=input_id.upper())
        '''
        input_title = product_df[product_df['asin'].str.lower() == input_id]['ori_title'].item()

        asin, title = topic_modeling_recommender(input_id, product_df)

        rec = 'Topic Modeling Recommender Result for {}, {}: <br>'.format(input_id.upper(), input_title)
        for i in range(0, 10):
            link = '<a class="" href="{}" target="_blank">{}</a>'.format(
                'https://www.amazon.com/dp/' + asin[i], asin[i])
            rec += '{}. {}, {}<br>'.format(i + 1, link, title[i])
        '''
    return rec


def topic_modeling_recommender(id, df, top_n=10):
    id = id.lower()

    # get the input product topic number
    topic_num = df[df['asin'].str.lower() == id]['topic_num'].item()

    # remove the input product from the recommendation data
    exclude_input_df = df.copy()
    exclude_input_df = exclude_input_df[exclude_input_df['asin'].str.lower() != id]

    # get the top 10 Probability product for the matching topic number
    output_df = exclude_input_df[exclude_input_df['topic_num'] == topic_num].sort_values('probability', ascending=False).head(top_n)

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
