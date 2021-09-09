# Starbucks Capstone Challenge

This project is a key part of Udacity Data Science Nanodegree program.

### Table of Contents

1. [Project Overview](#intro)
2. [Project Components](#components)
3. [Installing Dependencies](#installation)
4. [Directory Structure](#files)
5. [Instructions](#instructions)
6. [Modeling Offer Response](#results)
7. [Acknowledgements](#ack)


### 1. Project Overview<a name="intro"></a>

Starbucks regularly sends out offers to customers via its mobile app. An offer can be just an advertisement for a drink, i.e. informationl or an actual offer for customers to get a discount or BOGO (buy one get one free). Not all users receive the same offer, and not all customers receiving the offer respond to it. 

There are many areas that we can explore and analyze the pattern of Starbucks customer behavior. In this project, the main focus is to model whether a customer would respond to an offer based on the offer and customer's demographic information. An accurate model can help Starbucks to become more effective on sending offers to the right groups.

Three simulated datasets were provided for this challenge. These datasets mimics Starbucks' customer behavior using the Starbucks rewards mobile app.

A detailed write up of the project can be found at [Coffe or no coffe ?](https://github.com/Juliaj/udacity-data-scientist/blob/gh-pages/starbucks-challenge.md). 

### 2. Project Components<a name="components"></a>

Project contains following components.  

- [Exploring and cleaning the given data sets.](./notebooks/DataCleaningandProcessing.ipynb)
- [Modeling Offer Response and hyperparameter tuning.](./notebooks/ModelingOfferResponse.ipynb)  
- [Python modules for etl and modeling pipelines.](./pipelines) 
 
### 3. Installation<a name="installation"></a>

The project runs within an enviroment set up by pipenv. To install dependencies,

```
$ pip install pipenv  # if you haven't already
$ cd pipelines

# Mac osx users
$ export SYSTEM_VERSION_COMPAT=1

$ pipenv install
```

Notes:
- A requirements.txt file is provided for users who prefer to use pip. 
- In case there is an issue to install pytorch, run following 

```
$ pipenv shell
$ export SYSTEM_VERSION_COMPAT=1
$ pipenv install --verbose "https://download.pytorch.org/whl/cpu/torch-1.9.0-cp39-none-macosx_10_9_x86_64.whl"

```

### 4. Directory Structure<a name="files"></a>

    ├── data                      <-  the data repository.
    │   ├── 0_raw                 <-  containing raw datasets provided by Udacity. 
    │   │   ├── portfolio.json        <-  containing offer ids and meta data about each offer (duration, type, etc.)
    │   │   ├── profile.json          <-  demographic data for each customer
    │   │   └── transcript.json       <-  records for transactions and offers activities.
    │   │
    │   ├── 1_interim             <-  datasets produced from data cleaning and pre-processing.
    │   │   ├── offer_response.pkl    <-  a combined dataset based on portfolio, profile and transcript.
    │   │   ├── offer_summary.pkl     <-  a dataset to summarize response successs rate for offers.
    │   │   ├── portfolio.pkl         <-  a cleaned dataset based on portfolio.
    │   │   ├── profile.pkl           <-  a cleaned dataset of demographic info for each customer
    │   │   └── transcript.pkl        <-  a cleaned dataset based on transcript.json
    │   │
    ├── notebooks                 <- The folder for Jupyter notebooks.
    │   ├── Data Cleaning and Processing.ipynb    <-  Details for data cleaning and analysis.
    │   ├── ModelingOfferResponse.ipynb           <-  Model offer response.
    │   └── Starbucks_Capstone_notebook.ipynb     <-  Capstone challenge description.
    │
    ├── output                    <- folder for various artifacts.  
    │   └── models                    <-  folder for trained models.
    │       └── offer_response.pkl    <-  trained classifier.
    │
    ├── pipelines                 <- python modules for etl and modeling pipelines
    │   ├── data_processing
    │   │    ├── offer_response.py    <-  module to produce a combined dataset. 
    │   │    ├── portfolio.py         <-  module to clean portfolio data
    │   │    ├── profile.py           <-  module to clean profile data
    │   │    ├── transcript.py        <-  module to clean transcript data
    │   │    └── util.py              <-  shared utilities
    │   │
    │   ├── offer_response        <- modules to produce offer response model.
    │   │    ├── features.py          <-  module to create features.
    │   │    ├── model.py             <-  model training and evaluation 
    │   │    └── visualize.py         <-  helpful function to plot charts
    │   │
    │   ├── etl.sh                <- bash script to run all code within pipelines 
    │   │
    │   │
    ├── Pipfile                   <- pipevn configuration file 
    │
    ├── requirements.tx           <- configuration file to install dependencies via pip
    │           
    └── README.md                 <- A readme file 

### 5. Instructions<a name="instructions"></a>

1. The detailed anlaysis and modeling work are contained within the jupyter notebooks. To run them

```
$ pipenv shell
$ jupyter-lab

```

2. To run bash script for entire etl and modeling pipelines 

```
$ cd pipelines
$ pipenv shell

$ ./etl.sh
```

### 6. Modeling Offer Response

Two version of models were trained with sklearn RandomForestClassifier to predict whether a customer will respond to an offer given demagraphic and offer information. Both have good results. 
- v1 trained with a feature `purchase_in_offer` which captures the transactions that customer make during the offer period. f1-score was nearly perfect.
- v2 trained without this feature to become true predictive. f1-score was .67.

In both cases, f1-score was used to measure model performance because the dataset was balanced. 

Detailed evaluation using f1-score and confusion metrics can be found in [Modeling Offer Response](./notebooks/ModelingOfferResponse.ipynb). 

### 7. Acknowledgements<a name="ack">

The dataset used in this project contains simulated data from Starbucks rewards program. The data were made available via [Udacity Data Science Nanodegree](https://classroom.udacity.com/nanodegrees/nd025/dashboard/overview). 

This git repository was generated using [Carmine Paolino, cookiecutter-modern-datascience](https://github.com/crmne/cookiecutter-modern-datascience).
