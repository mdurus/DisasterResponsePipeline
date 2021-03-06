import sys
import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(messages_filepath, categories_filepath):
     
    """
    Function: Load the message and categorie db
    Args:
      messages_filepath(str): Path of messages database
      categories_filepath(str): Path of categories database
    Return:
      df(dataframe): Database that is merged of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how ='inner',on="id")
    return df


def clean_data(df):
    
    """
    Function: Formatt dataframe and split the categories and add as a dummy variable
    Args:
      df(dataframe): Db created by the merging of messages and categories      
    Return:
      df(dataframe): Formatted dataframe with the dummy variables for categories
    """
    
    categories = df.categories.str.split(';', expand=True)
    
    row = categories.iloc[[1]]
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    
    categories.columns = category_colnames
    
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    not_binary_columns=(categories.max()>1)[categories.max()>1].index
    # update non binary values to 1
    for col in not_binary_columns:
        #print(categories[col].value_counts())
        categories.loc[categories[col]>1,col] = 1
        #print(categories[col].value_counts())
    
    df = df.drop(['categories'],axis = 1)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(inplace = True)
    return df


def save_data(df, database_filename):
    
    """
    Function: Save the output dataframe as a SQL table
    Args:
      df(dataframe): Db created by the merging of messages and categories      
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')  


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