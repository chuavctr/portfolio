# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:22:26 2022

@author: vcmc4
"""

import time
import tweepy
import pandas as pd
import string
#from datetime import datetime
from dateutil.parser import parse
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from keybert import KeyBERT
from pygooglenews import GoogleNews

# load the model and vectorizer from respective file in given filename/directory
svc_model = pickle.load(open('linsvc_model_m8-2.pkl', 'rb'))
tfidf_vector = pickle.load(open('tfidf_vectorizer_linsvc.pkl', 'rb'))

#make pipeline of tfidf and logistic regression model
svc_tfidf_pipe = make_pipeline(tfidf_vector, svc_model)

#KeyBert model for text extraction
model = KeyBERT(model = "princeton-nlp/sup-simcse-bert-base-uncased")

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_KEY = ''
ACCESS_SECRET = ''

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)


#define a CleanText class to perform data cleaning on tweet text data from data set
class CleanText(BaseEstimator, TransformerMixin):
    #replacing the shorten word "re" with "reply"
    #def replace_re(self, input_text):
        # counter = 1
        # if counter ==1:
        #     print(counter)
        #     print(re.sub(r'\bre\b', "reply", input_text))
        #return re.sub(r'\bre\b', "reply", input_text)
    
    #remove all links from tweet 
    def remove_links(self, input_text):
        return re.sub(r'https?:\/\/\S+', '', input_text)
    
    #remove all mentions from tweet 
    def remove_mentions(self, input_text):
        return re.sub(r'@[A-Za-z0-9]+','', input_text)
    
    #remove all retweets from tweet 
    def remove_RT(self, input_text):
        return re.sub(r'RT[\s]+','', input_text)
    
    #remove all hashtags from tweet 
    def remove_hash(self, input_text):
        return re.sub(r'#','', input_text)
    
    #remove all punctuation from tweet 
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    
    #remove all digit from tweet text data
    #def remove_digits(self, input_text):
        #return re.sub('\d+', '', input_text)
    
    #change all letter in tweet text data to lowercase
    def to_lower(self, input_text):
        return input_text.lower()
    
    #remove of leading/trailing whitespace 
    def remove_space(self, input_text):
        return input_text.strip()
    
    #remove extra spaces
    def remove_extra_space(self, input_text):
        return re.sub(r'\s+', ' ', input_text, flags=re.I)
    
    #remove special character
    def remove_special_char(self, input_text):
        return re.sub(r'\W', ' ' , input_text)
    
    #remove all stopwords such as to, i, me, etc. from tweet
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words)

    
    #perform lemmatization on each word in tweet to return each word back to its root word without changing the meaning of the word
    def word_lemmatization(self, input_text):
        lemmatizer = WordNetLemmatizer()
        words = input_text.split()
        lemmed_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmed_words)
     

    def fit(self, X, y=None, **fit_params):
        return self

    #apply each function in CleanText class to the text dataframe
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_links).apply(self.remove_mentions).apply(self.remove_RT).apply(
            self.remove_hash).apply(self.remove_punctuation).apply(
            self.to_lower).apply(self.remove_space).apply(self.remove_extra_space).apply(
            self.remove_special_char).apply(self.remove_stopwords).apply(self.word_lemmatization) #.apply(self.remove_digits)
          
        return clean_X
ct = CleanText()

# Classification
def classify(text):
    classes = {}
    if ((svc_tfidf_pipe.predict_proba(text)[0,1] *100) < 50):
        catPct = svc_tfidf_pipe.predict_proba(text)[0,0] *100
        cat = "True"

    else:
        catPct = svc_tfidf_pipe.predict_proba(text)[0,1] *100
        cat = "False"
    
    classes['cat'] = cat
    classes['catPct'] = round(catPct, 2)
    return classes

# Keyword Extraction
def extract(text):
    words = model.extract_keywords(
                    text,
                    top_n=1,
                    keyphrase_ngram_range=(1,2),
                    stop_words="english",
                )
    words_Str = ' '.join([str(elem) for elem in words])
    syntax = re.compile("'[^']*'")
    extract = {}
    for value in syntax.findall(words_Str):
        extract = value
    return extract

# Keyword Query
def query(text):
    gn = GoogleNews(lang = 'en')
    news = {}
    syntax = re.compile("'[^']*'")
    for value in syntax.findall(text + "covid" ): # Adding covid literally helps to narrow searches down
        search = gn.search(value)
        newsitem = search['entries']
        news['title'] = newsitem[0].title
        news['link'] = newsitem[0].link

    return news

# Posting things to Twitter
def posting(tID, tName, cat, keyW, news):
    if bool(keyW): 
        tweetStatus = str( "@" + tName + " \nClassification: " + str(cat['cat']) + " ( Confidence: " + str(cat['catPct']) + "% ) " +  "\nKeywords: " + str(keyW) + "\nSuggested Article: \n" + str(news['title']) + "\n" + news['link'])
    else:
        tweetStatus = str( "@" + tName + " \nClassification: " + str(cat['cat']) + " ( Confidence: " + str(cat['catPct']) + "% ) " +  "\nKeywords: None Found!" )
    # Post Tweet
    print("Replying Tweet: " + str(tID))
    #api.update_status(status = tweetStatus, in_reply_to_status_id = tID) 

def main():
    
    try:
        # Retrieve Twitter Data
        api = tweepy.API(auth)
        mentions = api.mentions_timeline()
        
        # Read existing CSV
        db = pd.read_csv("tweetset.csv")
        
        dbTime = parse(db.datetime[0]) # 0 by default
        
        data = {'username': [],
                'datetime': [],
                'text':[],
                'tweet_id': [],
                'reply_id': [],
                'reply_text': [],
                'response': [],
                }
        
        df = pd.DataFrame(data)
        
        user_reply = []
        
        # Retrieve tweets that only contain keyword & is replying to a tweet:
        # AFTER A CERTAIN DATETIME
        
        try:
            for count, i in enumerate(mentions):
                datetime = i.created_at
                if "getInfo" in i.text:
                    if  datetime > dbTime:
                        if isinstance(i.in_reply_to_status_id, int):
                                username = i.user.screen_name
                                text = i.text
                                tweet_id = i.id
                                reply_id = i.in_reply_to_status_id
                                reply_text = api.get_status(i.in_reply_to_status_id).text
                                response = "false"
                                df.loc[count] = [username, datetime,text,tweet_id, reply_id, reply_text, response]
                                user_reply = api.get_status(i.in_reply_to_status_id).text
        
        except Exception as e:
            print(e)
        
        # Extract Unresponded Tweets, otherwise end program (Check if need to respond)
        if "false" in df['response'].unique():
            unres = df[df['response'].str.contains('false')]
        else:
            print("No new tweets detected!")
            
            return
        
        resp = pd.DataFrame()
        
        #Function based:
        for i in range(len(unres)):
            if "false" in unres.response.iloc[i]: # if not responded: (NO NEED AFTER TEST, unres is all false)
                if isinstance(unres.reply_id.iloc[i], np.float64): # is a reply
                    user_reply = unres.reply_text.iloc[i]
                    user_reply_clean = ct.transform(pd.Series([user_reply])) # Clean original Tweet
                        
                    # Classification
                    cat = classify(user_reply_clean)
                        
                    # Keyword Extraction
                    keyW = extract(user_reply_clean[0])
                    
                    if bool(keyW): 
                        news = query(keyW)
                        
                    else:
                        print("No value")
                        news = {'title': 'No Value', 'link': 'No Value'}
                        
                    # Posting on Twitter  
                    tweetID = int(unres.tweet_id.iloc[i])
                    tName = str(unres.username.iloc[i])
                    posting(tweetID, tName, cat, keyW, news)
                        
                    info = {}
                    info['datetime'] = unres.datetime.iloc[i]
                    info['cat'] = cat['cat']
                    info['conf'] = cat['catPct']
                    info['extract'] = str(keyW)
                    info['nTitle'] = news['title']
                    info['nLink'] = news['link']
                    
                    # Adding info into dataframe
        
                    resp = resp.append(info, ignore_index = True)
                
                else:
                    print("This is not a reply!")
                    
                # Change Status to replied
                df.loc[df.datetime == unres.datetime.iloc[i], 'response'] = "True"
                print("Tweet Posted")
                
        # merge dataframe based on ID
        df = df.merge(resp, how = 'left', on='datetime')
        
        # add this to db dataframe
        db = pd.concat([df, db]).reset_index(drop=True)     
        
        #Save To CSV
        db.to_csv("tweetset.csv", index = False)
        
        print('code complete')
        
    except Exception as e:
        print(e)


try:
    while True:
        main()
        time.sleep(120)
except KeyboardInterrupt:
    print("Input Detected; ending program...")
    