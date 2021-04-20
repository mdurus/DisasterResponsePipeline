# Disaster Response Pipeline Project

Idea behind the project:
    Web app to be constructed where an emergency worker can input a new message and get classification results in several categories. With the help of this, they can easily analyze the messages to predict the disasters. App displays visualizations of the data also.
     
## 1. Used libraries;
    - Pandas
    - Numpy
    - nltk
    - sklearn
    - sqlalchemy

## 2. Files
    - app
    -   * template
        *   * master.html # main page of web app
        *   * go.html # classification result page of web app
    -   * run.py # Flask file that runs app
    - data
    -   * disaster_categories.csv # category data to process
    -   * disaster_messages.csv # message data to process
    -   * process_data.py # python file to run the ETL pipeline
    -   * DisasterResponse.db # database to save clean data to
    - models
    -   * train_classifier.py # python file to run the ML pipeline
    -   * classifier.pkl # saved model with 3 parts to pass 25mb size limit
    - README.md
    - ETL Pipeline Preparation.ipynb # 1st part of the project
    - ML Pipeline Preparation.ipynb # 2nd part of the project

## 3. Pipelines

Process_data.py and train_classifier.py files are constructed by using ipynb notebooks. Details of the steps are provided in the notebooks.

## 4. Flask Web App

To have a visualization a web app is used. To run our web app python run.py Run env | grep WORK command is used to obtain space-id and workspace-id.
You can reach the web app from 'https://view6914b2f4-3001.udacity-student-workspaces.com/'
    
![image](https://user-images.githubusercontent.com/26851673/115455489-b1307080-a22a-11eb-9f8e-5c177262480f.png)

