import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - (str) filepath to messages dataframe
    categories_filepath - (str) filepath to categories dataframe 
    
    OUTPUT:
    df - (pandas dataframe) messages dataframe merged whith categories dataframe  
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='outer', on='id') 
    return df



def clean_data(df):
    '''
    INPUT:
    df - (pandas dataframe) messages dataframe merged whith categories dataframe
    
    OUTPUT:
    df_clean - (pandas dataframe) cleaned dataframe: categories column replaced by
        separate columns for each category with 0's and 1's, duplicates dropped 
    
    '''
    # separate the categories 
    categories = df.categories.str.split(pat=';', expand=True)
    
    # extract a list of new column names for categories
    row = categories.loc[0,:]
    category_colnames = row.str.split(pat='-', expand=True)[0]
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat='-', expand=True)[1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('float')
    
    # turn all 2's into 1's for 'related' column
    categories.related = categories.related.replace({2:1})
    # drop the original categories column from `df`
    df = df.drop(labels='categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df_clean = pd.concat([df, categories], axis=1)

    # drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def save_data(df, database_filename):
    '''
    INPUT:
    df - (pandas dataframe) cleaned dataframe
    database_filename - (str) filename for sqlite database to store cleaned data 
    
    OUTPUT:
    None
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('cleaned_data', engine, index=False)
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()