import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import pickle


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - (str) filepath to sqlite database
    
    OUTPUT:
    X - (pandas series) series with text messages
    Y - (pandas dataframe) dataframe with 36 categories
    category_names - (list) category names
    
    '''
    # read data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_data', engine)
    
    # extract the variables 
    X = df.message
    Y = df.drop(labels=['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)
    
    return X, Y, category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    '''
    INPUT:
    text - (str) raw text for cleaning
    
    OUTPUT:
    clean_tokens - (list) lemmatized and case normalized tokens
    
    '''
    # replace url's with placeholders
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize and case normalize 
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT:
    None
    
    OUTPUT:
    cv - (model) TF-IDF pipeline with gridsearch
    
    '''
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # define gridsearch parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'clf__estimator__n_estimators': [50, 100, 200]
    }

    # add gridsearch to the model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - (model) TF-IDF pipeline with gridsearch
    X_test - (pandas series) series with text messages
    Y_test - (pandas dataframe) dataframe with 36 categories
    category_names - (list) category names
    
    OUTPUT:
    None
    
    '''
    # get the predictions
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=Y_test.columns, index=Y_test.index)
    
    # print the f1 score, precision and recall for each category
    for column in category_names:
        print(classification_report(Y_test[column], y_pred[column]))
        
    return None


def save_model(model, model_filepath):
    '''
    INPUT:
    model - (model) TF-IDF pipeline with gridsearch
    model_filepath - (str) filepath to save the model 
    
    OUTPUT:
    None
    
    '''
    # open a file, where you ant to store the data
    file = open(model_filepath, 'wb')

    # dump information to that file
    pickle.dump(model, file)

    # close the file
    file.close()
    
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()