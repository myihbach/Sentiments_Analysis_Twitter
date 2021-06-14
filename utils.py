import re
import string
import csv
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from string import punctuation


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    lemmatizer = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
                
            #stem_word = stemmer.stem(word)  # stemming word
            #tweets_clean.append(stem_word)
            lemm_word = lemmatizer.lemmatize(word)

            tweets_clean.append(lemm_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


def remove_punctuation(s):
    return ''.join(c for c in s if c not in '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~' )


def abbreviation(tweet):
    user_string = remove_punctuation(tweet)
    user_string = user_string.split(" ")
    
    j = 0
    for _str in user_string:
        fileName= "slangs.csv"
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            dataFromFile = csv.reader(myCSVfile,delimiter=",")
            #_str = _str.translate(str.maketrans('','',string.punctuation))
            for row in dataFromFile:
                if _str.upper() == row[1]:
                    user_string[j] = row[2]
            myCSVfile.close()
        j = j + 1
    return ' '.join(user_string)
