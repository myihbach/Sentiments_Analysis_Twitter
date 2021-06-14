from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
from gensim.models import Word2Vec
from utils import process_tweet


app = Flask(__name__)
#Machine Learning code goes here


w2v = Word2Vec.load('models/word2vec_tweets.model')
model = pickle.load(open('models/Logistic_Regression.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


def tweet_to_vec(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tweet:
        try:
            vec += w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        tweet = request.form['tweet']
        preprocessed_tweet = process_tweet(tweet)
        tweet2vec = tweet_to_vec(tweet, 100)
        prediction = model.predict(tweet2vec)
    return render_template('result.html', prediction=prediction )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)