import os
import pathlib
#import tarfile
import urllib.request
import pandas as pd
import spacy
import string
#import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pickle import dump, load
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


    
#Custom transformer using Python standard library (you could use spacy as well)
class predictors(TransformerMixin):

    # This function will clean the text
    def clean_text(self,text):     
        return text.strip().lower()

    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]
        #return [text for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
        

class SentimentTrain(object):

    def __init__(self,data_path):
        self.data_path=os.path.join(pathlib.Path().absolute(), data_path)

    def prepareData(self):

        df_yelp = pd.read_table(os.path.join(self.data_path,'yelp_labelled.txt'))
        df_imdb = pd.read_table(os.path.join(self.data_path,'imdb_labelled.txt'))
        df_amz = pd.read_table(os.path.join(self.data_path,'amazon_cells_labelled.txt'))
        # Concatenate our Datasets
        frames = [df_yelp,df_imdb,df_amz]

        for column in frames: 
            column.columns = ["Message","Target"]

        df = pd.concat(frames)
        return df

    def spacy_tokenizer(self,doc):
        punctuations = string.punctuation
        nlp = spacy.load('en_core_web_sm')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS

        tokens = nlp(doc)
        # Lemmatizing each token and converting each token into lowercase
        tokens = [word.lemma_.lower() for word in tokens]        
        # Removing stop words and punctuations
        tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]
        # return preprocessed list of tokens
        return tokens

    def train(self):
        df = self.prepareData()

        tfvectorizer = TfidfVectorizer(tokenizer = self.spacy_tokenizer)
        classifier_LG = LogisticRegression(verbose=True)

        pipe2_LG = Pipeline([
            ("cleaner", predictors()),
            ('vectorizer', tfvectorizer),
            ('classifier', classifier_LG)], verbose=True)

        # pipe2_LG = Pipeline([
        #     ("cleaner", predictors()),
        #     ('vectorizer', tfvectorizer)], verbose=True)

        X = df['Message']
        ylabels = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

        pipe2_LG.fit(X_train,y_train)
        #pipe2_LG.fit_transform(X_train,y_train)

        # Save the model
        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/logreg_tfidf.pkl"

        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        dump(pipe2_LG, open(model_file, 'wb'))

        #return self.spacy_tokenizer(frames[0]["Message"][0])
        #return predictors().fit(X_train)
        #tf = tfvectorizer.fit_transform(X_train)
        example = ["I do enjoy my job",
        "What a poor product!,I will have to get a new one",
        "I feel amazing!",
        "This class sucks"]
        #pred = predictors().fit_transform(example)
        
        #return pipe2_LG.fit_transform(example).toarray()
        return pipe2_LG


class PredictSentiment(object):

    def __init__(self):
        #model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        #model_file = model_path + "/forest_reg.pkl"
        #self.model = load(open(model_file, 'rb'))
        self.model = joblib.load("model/logreg_tfidf.pkl")

    def predict(self, sentence):

        predict = self.model.predict(sentence)

        return predict
        #return sentence











