"""
A demo performing sentiment analysis on random Tweets matching a particular keyword

Source: https://www.youtube.com/watch?v=o_OZdbCzHUA&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=2
"""
import tweepy

import textblob

consumer_key = "XXXX"
consumer_secret = "XXXX"

access_token = "XXXX"
access_token_secret = "XXXX"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Arsenal')

for tweet in public_tweets:
    print(tweet.text)
    analysis = textblob.TextBlob(tweet.text)
    print(analysis.sentiment)
