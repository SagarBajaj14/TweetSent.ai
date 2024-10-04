import pickle
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st

nltk.download('wordnet')

def load_models():
    with open('TFIDF/vectoriser-ngram-(1,2).pickle', 'rb') as file:
        vectoriser = pickle.load(file)

    with open('TFIDF/Sentiment-LR.pickle', 'rb') as file:
        LRmodel = pickle.load(file)
    

    return vectoriser, LRmodel

def preprocess(textdata):
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    processedText = []
    wordLemm = WordNetLemmatizer()

    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern, ' URL', tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        tweet = re.sub(userPattern, ' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
    
        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')

        processedText.append(tweetwords)  

    return processedText

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess([text]))
    sentiment = model.predict(textdata)

    df = pd.DataFrame([(text, sentiment[0])], columns=['text', 'sentiment'])
    df['sentiment'] = df['sentiment'].replace([0, 1], ["Negative", "Positive"])
    return df

def main():
    st.title("Twitter Sentiment Analysis")
    
    vectoriser, LRmodel = load_models()

    user_input = st.text_area("Enter a tweet:", height=100)

    if st.button("Predict Sentiment"):
        if user_input:
            result_df = predict(vectoriser, LRmodel, user_input)
            st.write(result_df)
        else:
            st.warning("Please enter a tweet for sentiment analysis.")

if __name__ == "__main__":
    main()
