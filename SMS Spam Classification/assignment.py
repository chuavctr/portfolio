# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:59:36 2021

@author: vcmc4
"""

import numpy as np 
import pandas as pd
import streamlit as st
import string
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

from sklearn.preprocessing import MinMaxScaler
from random import randint
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
#import wordcloud

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

@st.cache(suppress_st_warning=True)
def load_data():
    data = pd.read_csv('spam.csv',encoding='latin-1')
    data = data.dropna(axis='columns')
    data = data.rename(columns= {"v1":"Label"})
    data = data.rename(columns= {"v2":"Text"})
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

def remove_punctuation_and_stopwords(sms):
    
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    
    sms_no_punctuation_no_stopwords = \
        [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
        
    return sms_no_punctuation_no_stopwords

st.title("SMS Text Spam")

st.header("By: Victor (0129219) & Joseph (0125112)")

st.subheader("Explain Classification topic... ")

data = pd.read_csv("spam.csv", encoding = 'latin-1')
data = data.dropna(axis='columns')
data = data.rename(columns= {"v1":"Label"})
data = data.rename(columns= {"v2":"Text"})
nav = st.sidebar.radio("Contents", ["Home", "Details"])




if nav == "Home":
    st.subheader("List of SPAM & HAM messages")
    def data():
        num = st.number_input("Enter how many records to view?", min_value=1, max_value=None, step=1)
        data_load_state = st.text('Loading data...')
        data = load_data()
        data_load_state.text("Done! (using st.cache)")
        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(data.iloc[:num])
        return data
    
    cdf = data()
    
if nav =="Details":
    graph = st.selectbox("What kind of graph?", ["General Statistics","Data Comparison" , "Numerical Feature Extraction" ,"Models comparison", "LIME"])
    data['spam'] = data['Label'].map({'spam': 1, 'ham': 0}).astype(int)
    data['length'] = data['Text'].apply(len)
    
    #data['length'] = data['Text'].apply(len)  
    data_ham  = data[data['spam'] == 0].copy()
    data_ham.loc[:, 'Text'] = data_ham['Text'].apply(remove_punctuation_and_stopwords)
    words_data_ham = data_ham['Text'].tolist()
    list_ham_words = []
    for sublist in words_data_ham:
        for item in sublist:
            list_ham_words.append(item)
    c_ham  = Counter(list_ham_words)
    df_hamwords_top30  = pd.DataFrame(c_ham.most_common(30),  columns=['word', 'count'])
    
    data_spam = data[data['spam'] == 1].copy()
    data_spam.loc[:, 'Text'] = data_spam['Text'].apply(remove_punctuation_and_stopwords)
    words_data_spam = data_spam['Text'].tolist()
    list_spam_words = []
    for sublist in words_data_spam:
        for item in sublist:
            list_spam_words.append(item)
    c_spam = Counter(list_spam_words)
    df_spamwords_top30 = pd.DataFrame(c_spam.most_common(30), columns=['word', 'count']) 
    
    bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(data['Text'])
    #Applying bow_transformer on all messages
    bow_data = bow_transformer.transform(data['Text'])
    tfidf_transformer = TfidfTransformer().fit(bow_data)
    data_tfidf = tfidf_transformer.transform(bow_data)
    data_tfidf_train, data_tfidf_test, label_train, label_test = \
        train_test_split(data_tfidf, data["spam"], test_size=0.3, random_state=5)
    
    X2 = hstack((data_tfidf ,np.array(data['length'])[:,None])).A
    X2_train, X2_test, y2_train, y2_test = \
        train_test_split(X2, data["spam"], test_size=0.3, random_state=5)
    
    if graph == "General Statistics":
        st.header("Ham vs Spam - Statistics")
        st.subheader("Ham VS Spam: Statistics Description")
        st.write(data.groupby("Label").describe())
        st.subheader("Ham VS Spam Message Count")
        fig = sns.factorplot(x="Label", data=data, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")
        st.pyplot(fig)
        
    if graph == "Data Comparison":
        

    
        st.header("Ham VS Spam - Statistics")
        st.subheader("Ham VS Spam: Word Count per Message")
        fig = data.hist(column='length', by='Label', bins=60, figsize=(12,4))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.show()
        st.pyplot()
        
        st.header("Top 30 words")
        
        st.subheader("Top 30 Ham Words")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='word', y='count', 
        data=df_hamwords_top30, ax=ax)
        plt.title("Top 30 Ham words")
        plt.xticks(rotation='vertical');
        st.pyplot()
        
    if graph == "Numerical Feature Extraction":
        def list_words(cols):
            for col in cols: 
                st.write(bow_transformer.get_feature_names()[col])
            return data
        
        st.title("Numerical Feature Extraction")
        bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(data['Text'])
        st.header("Bag of Words (BoW) Text Vectorization")
        st.write('In all of the messages, bow_transformer has counted ', len(bow_transformer.vocabulary_), " unique words.")
        
        st.header("Examples for spam and ham messages (Vectorization Examples)")
        st.subheader("Spam Messages")
        sample_spam = data['Text'][8]
        bow_sample_spam = bow_transformer.transform([sample_spam])
        st.subheader("SAMPLE SPAM: \n")
        st.write(sample_spam)
        
        rows, cols = bow_sample_spam.nonzero()
        st.subheader("List of spam feature names:")
        list_words(cols)
        
            
        st.subheader("Ham Messages")
        sample_ham = data['Text'][4]
        bow_sample_ham = bow_transformer.transform([sample_ham])
        st.subheader("SAMPLE HAM: \n")
        st.write(sample_ham)
        
        rows, cols = bow_sample_ham.nonzero()
        st.subheader("List of ham feature names:")
        list_words(cols)

        
        #No of none zero entries divided by matrix size
        st.subheader("Percentage of non zeroes in the matrix")
        st.write(((bow_data.nnz / (bow_data.shape[0] * bow_data.shape[1]) *100) *100), "% of the matrix are non zeroes (ones)") 
        
    if graph == "Models comparison":
        sms_train, sms_test, label_train, label_test = \
            train_test_split(data["Text"], data["spam"], test_size=0.3, random_state=5)
        
        
        st.title("Comparison of Classifer Models")
        # K Nearest Neighbors
        
        
        # Naive Bayes 
        data_tfidf_train = data_tfidf_train.A
        data_tfidf_test = data_tfidf_test.A
        
        spam_detect_model_minmax = MultinomialNB().fit(data_tfidf_train, label_train)
        pred_test_MNB = spam_detect_model_minmax.predict(data_tfidf_test)
        acc_MNB = accuracy_score(label_test, pred_test_MNB)
        f1_MNB = f1_score(label_test, pred_test_MNB)
        rec_MNB = recall_score(label_test, pred_test_MNB)
        prec_MNB = precision_score(label_test, pred_test_MNB)
        st.header("Naive Bayes ")
        st.write("Naive Bayes Scores:")
        st.write("Accuracy: ", acc_MNB)
        st.write("F1 Score: ", f1_MNB) 
        st.write("Recall Score: ", rec_MNB)
        st.write("Precision Score: ", prec_MNB)
        nb = acc_MNB
        
        scaler = MinMaxScaler()
        data_tfidf_train_sc = scaler.fit_transform(data_tfidf_train)
        data_tfidf_test_sc  = scaler.transform(data_tfidf_test)

        #NB - TFIDF Matrix
        spam_detect_model_minmax = MultinomialNB().fit(data_tfidf_train_sc, label_train)
        pred_test_MNB = spam_detect_model_minmax.predict(data_tfidf_test_sc)
        acc_MNB = accuracy_score(label_test, pred_test_MNB)
        f1_MNB = f1_score(label_test, pred_test_MNB)
        rec_MNB = recall_score(label_test, pred_test_MNB)
        prec_MNB = precision_score(label_test, pred_test_MNB)
        st.header("Naive Bayes (TFIDF Matrix)")
        st.write("Naive Bayes (TFIDF Matrix) Scores:")
        st.write("Accuracy: ", acc_MNB)
        st.write("F1 Score: ", f1_MNB) 
        st.write("Recall Score: ", rec_MNB)
        st.write("Precision Score: ", prec_MNB)
        nb_TFIDF = acc_MNB
       
        #NB with TFIDF Matrix + Feature Length (Unscaled) [VALUES ARE STILL WRONG SO PLEASE CHECK THIS PART OUT]
        spam_detect_model_2 = MultinomialNB().fit(X2_train, y2_train)
        pred_test_MNB_2 = spam_detect_model_2.predict(X2_test)
        acc_MNB_2 = accuracy_score(y2_test, pred_test_MNB_2)
        f1_MNB_2 = f1_score(y2_test, pred_test_MNB_2)
        rec_MNB_2 = recall_score(y2_test, pred_test_MNB_2)
        prec_MNB_2 = precision_score(y2_test, pred_test_MNB_2)
        st.header("Naive Bayes (TFIDF Matrix) + Feature 'Length' (Unscaled)")
        st.write("Naive Bayes (TFIDF Matrix) + 'Length' (Unscaled) Scores:")
        st.write("Accuracy: ", acc_MNB_2)
        st.write("F1 Score: ", f1_MNB_2) 
        st.write("Recall Score: ", rec_MNB_2)
        st.write("Precision Score: ", prec_MNB_2)
        
        #Scaling
        X2_tfidf_train = X2_train[:,0:9431]
        X2_tfidf_test  = X2_test[:,0:9431]
        X2_length_train = X2_train[:,9431]
        X2_length_test  = X2_test[:,9431]
        
        scaler = MinMaxScaler()
        X2_length_train = scaler.fit_transform(X2_length_train.reshape(-1, 1))
        X2_length_test  = scaler.transform(X2_length_test.reshape(-1, 1))
        
        X2_train = np.hstack((X2_tfidf_train, X2_length_train))
        X2_test  = np.hstack((X2_tfidf_test,  X2_length_test))
        
        #Naive Bayes with TFIDF Matrix + Feature Length (Scaled) 
        spam_detect_model_3 = MultinomialNB().fit(X2_train, y2_train)
        pred_test_MNB_3 = spam_detect_model_2.predict(X2_test)
        acc_MNB_3 = accuracy_score(y2_test, pred_test_MNB_3)
        f1_MNB_3 = f1_score(y2_test, pred_test_MNB_3)
        rec_MNB_3 = recall_score(y2_test, pred_test_MNB_3)
        prec_MNB_3 = precision_score(y2_test, pred_test_MNB_3)
        st.header("Naive Bayes (TFIDF Matrix) + Feature 'Length' (Scaled)")
        st.write("Naive Bayes (TFIDF Matrix) + 'Length' (Scaled) Scores:")
        st.write("Accuracy: ", acc_MNB_3)
        st.write("F1 Score: ", f1_MNB_3) 
        st.write("Recall Score: ", rec_MNB_3)
        st.write("Precision Score: ", prec_MNB_3)
        
        #comparison graph for all score models?
        st.header("Classifier Pipelines")
        #K Nearest Neighbors
        st.subheader("K Nearest Neighbors (KNN)")
        
        parameters_KNN = {'n_neighbors': (10,15,17), }
        grid_KNN = GridSearchCV( KNeighborsClassifier(), parameters_KNN, cv=5,
                                n_jobs=-1, verbose=1)
        grid_KNN.fit(data_tfidf_train, label_train)
        st.write("KNN Scores - Accuracy: ", grid_KNN.best_score_)
        
        #Multinomial NB 
        #simple pipeline. no optimization
        pipe_MNB = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                           ('tfidf'   , TfidfTransformer()),
                           ('clf_MNB' , MultinomialNB()),])
        pipe_MNB.fit(X=sms_train, y=label_train)
        pred_test_MNB = pipe_MNB.predict(sms_test)
        acc_MNB = accuracy_score(label_test, pred_test_MNB)
        f1_MNB = f1_score(label_test, pred_test_MNB)
        rec_MNB = recall_score(label_test, pred_test_MNB)
        prec_MNB = precision_score(label_test, pred_test_MNB)
        st.header("Multinomial NB ")
        st.write("Multinomial NB Scores:")
        st.write("Accuracy: ", acc_MNB)
        st.write("F1 Score: ", f1_MNB) 
        st.write("Recall Score: ", rec_MNB)
        st.write("Precision Score: ", prec_MNB)
        
        #Naive Bayes
        pipe_MNB_tfidfvec = Pipeline([ ('tfidf_vec' , TfidfVectorizer(analyzer = remove_punctuation_and_stopwords)),
                               ('clf_MNB'   , MultinomialNB()),])
        pipe_MNB_tfidfvec.fit(X=sms_train, y=label_train)
        pred_test_MNB_tfidfvec = pipe_MNB_tfidfvec.predict(sms_test)
        acc_MNB_tfidfvec = accuracy_score(label_test, pred_test_MNB_tfidfvec)
        f1_MNB_tfidfvec = f1_score(label_test, pred_test_MNB_tfidfvec)
        rec_MNB_tfidfvec = recall_score(label_test, pred_test_MNB_tfidfvec)
        prec_MNB_tfidfvec = precision_score(label_test, pred_test_MNB_tfidfvec)
        st.header("Multinomial NB (TFIDF)")
        st.write("Multinomial NB (TFIDF) Scores:")
        st.write("Accuracy: ", acc_MNB_tfidfvec)
        st.write("F1 Score: ", f1_MNB_tfidfvec) 
        st.write("Recall Score: ", rec_MNB_tfidfvec)
        st.write("Precision Score: ", prec_MNB_tfidfvec)
        
        #SVM 
        pipe_SVC = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_SVC' , SVC(gamma='auto', C=1000)),
                    ])
        parameters_SVC = dict(tfidf=[None, TfidfTransformer()],
                      clf_SVC__C=[500, 1000,1500]
                      )
        grid_SVC = GridSearchCV(pipe_SVC, parameters_SVC, 
                        cv=5, n_jobs=1, verbose=1)
        grid_SVC.fit(X=sms_train, y=label_train)
        pred_test_SVC = grid_SVC.predict(sms_test)
        acc_SVC = accuracy_score(label_test, pred_test_SVC)
        f1_SVC = f1_score(label_test, pred_test_SVC)
        rec_SVC = recall_score(label_test, pred_test_SVC)
        prec_SVC = precision_score(label_test, pred_test_SVC)
        st.header("Support Vector Machine (SVM)")
        st.write("SVM Scores:")
        st.write("Accuracy: ", acc_SVC)
        st.write("F1 Score: ", f1_SVC) 
        st.write("Recall Score: ", rec_SVC)
        st.write("Precision Score: ", prec_SVC)
        
        #LIME
        class_names=['Ham','Spam']
        explainer = LimeTextExplainer(class_names=class_names)
        
        idx = 2
        exp = explainer.explain_instance(data.Text[idx], pipe_MNB_tfidfvec.predict_proba, num_features=15)
        st.write('Document id: %d' % idx)
        st.write('Probability=', pipe_MNB_tfidfvec.predict_proba([data.Text[idx]])[0,1])
        st.write('True class: %s' % data.Label[idx])
        exp_html = components.html(exp.as_html(), height = 400, width = 1000)
        
        fig = exp.as_pyplot_figure()
        
        st.pyplot(fig)
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(13,6))
        cheong = confusion_matrix(pred_test_MNB_tfidfvec, label_test)
        sns.heatmap(cheong, annot = True, annot_kws={"size":16}, fmt = 'd')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.grid(False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
    if graph == "LIME":
        sms_train, sms_test, label_train, label_test = \
            train_test_split(data["Text"], data["spam"], test_size=0.3, random_state=5)
        pipe_MNB_tfidfvec = Pipeline([ ('tfidf_vec' , TfidfVectorizer(analyzer = remove_punctuation_and_stopwords)),
                               ('clf_MNB'   , MultinomialNB()),])
        pipe_MNB_tfidfvec.fit(X=sms_train, y=label_train)
        pred_test_MNB_tfidfvec = pipe_MNB_tfidfvec.predict(sms_test)
        
        button = st.button("Randomize")
        if button:
            class_names=['Ham','Spam']
            explainer = LimeTextExplainer(class_names=class_names)
            
            for idx in range(len(data)):
                idx = randint(0, len(data))
                   
            exp = explainer.explain_instance(data.Text[idx], pipe_MNB_tfidfvec.predict_proba, num_features=15)
            st.write('Document id: %d' % idx)
            st.write('Probability=', pipe_MNB_tfidfvec.predict_proba([data.Text[idx]])[0,1])
            st.write('True class: %s' % data.Label[idx])
            exp_html = components.html(exp.as_html(), height = 400, width = 1000)
            
            fig = exp.as_pyplot_figure()
        
    
#st.table(data)