# Disaster Response Pipeline Project


#### Table of Contents
1. [Summary](#summary)
2. [Instructions](#instructions)
3. [The files in the repository](#files)


### Summary:
In this project we build an ETL pipeline that takes the datasets for disaster messages and categories, cleans them, and stores the clean data into a SQLite database. We also build a machine learning pipeline that takes the data from the database and creates and trains a classifier, and stores the classifier into a pickle file. Lastly, we build a web app based on the classifier and the database that for a given message returns classifications for all 36 classes. The app also provides visualizations for message genre distribution, message category distribution and message length distribution.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### The files in the repository:
The repository contains the following files:

* app/run.py - The web app that for a given message returns classifications for all 36 classes. The app also provides visualizations for message genre distribution, message category distribution and message length distribution. 

* app/templates/master.html, app/templates/go.html - html files for the main page and classification pages of the app. 

* data/disaster_categories.csv - dataframe with categories for messages

* data/disaster_messages.csv - dataframe for messages

* data/process_data.py - The ETL script that takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.

* models/train_classifier.py - The machine learning script that takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
