# Starbucks Capstone Challenge

This project is a key part of Udacity Data Science Nanodegree program.

### Table of Contents

1. [Project Overview](#objectives)
2. [Project Components](#components)
3. [Installing Dependencies](#installation)
4. [Directory structure](#files)
5. [Instructions](#instructions)
6. [Modeling results](#results)
7. [Acknowledgements](#ack)


### 1. Project Objective<a name="objectives"></a>

Starbucks regularly sends out offers to customers via its mobile app. An offer can be just an advertisement for a drink, i.e. informationl or an actual offer for customers to get a discount or BOGO (buy one get one free). Not all users receive the same offer, and not all customers receiving the offer respond to it. There are many areas that we can explore and analyze the pattern of Starbucks customer behavior. In this project, the main focus is to model whether a customer would respond to an offer based on the offer and customer's demographic information. An accurate model can help Starbucks to become more effective on sending offers to the right groups.

Three simulated datasets were provided for this challenge. These datasets mimics Starbucks' customer behavior using the Starbucks rewards mobile app.

### 2. Project Components<a name="components"></a>

The solutions are provided as following components.  

1- [Exploring and cleaning the given data sets]()
2- [Modeling Offer Response]().  
3- [A set of python modules](). 
 
### 3. Installation<a name="installation"></a>

The project runs within an enviroment set up by pipenv. To install dependencies,

```
$ pip install pipenv  # if you haven't already
$ cd pipelines

# Mac osx users
$ export SYSTEM_VERSION_COMPAT=1

$ pipenv install
```
A requirements.txt file is also provided for users who prefer to use pip. 

### 4. Directory structure<a name="files"></a>

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
    ├── notebooks                  <- The folder for Jupyter notebooks.
    │   ├── Data Cleaning and Processing.ipynb    <-  Details for data cleaning and analysis.
    │   ├── ModelingOfferResponse.ipynb           <-  Model offer response.
    │   └── Starbucks_Capstone_notebook.ipynb     <-  Capstone challenge description.
    │
    ├── output                   <- folder for various artifacts.  
    │   └── models                   <-  folder for trained models.
    │       └── offer_response.pkl   <-  trained classifier.
    │
    ├── pipelines                <- python modules for etl and modeling pipelines
    │   ├── data_processing
    │   │    ├── offer_response.py    <-  module to produce a combined dataset. 
    │   │    ├── portfolio.py         <-  module to clean portfolio data
    │   │    ├── profile.py           <-  module to clean profile data
    │   │    ├── transcript.py        <-  module to clean transcript data
    │   │    └── util.py              <-  shared utilities
    │   │
    │   ├── offer_response       <- modules to produce offer response model.
    │   │    ├── features.py          <-  module to create features.
    │   │    ├── model.py             <-  model training and evaluation 
    │   │    └── visualize.py         <-  helpful function to plot charts
    │   │
    │   ├── etl.sh               <- bash script to run all code within pipelines 
    │   │
    │   │
    ├── Pipfile                  <- pipevn configuration file 
    │
    ├── requirements.tx          <- configuration file to install dependencies via pip
    │
    └── README.md

### 5. Instructions<a name="instructions"></a>

1. The entire anlaysis is contained within the jupyter notebooks.
2. To run bash script for etl and modeling pipelines 

```
$ cd pipelines
$ pipenv shell

$ ./etl.sh
```

### 6. Modeling for Offer Response


### 7. Acknowledgements<a name="ack">

This project was completed as part of the [Udacity Data Science Nanodegree]. The dataset used in this project contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. [Starbucks® Rewards program: Starbucks Coffee Company](https://www.starbucks.com/rewards/).
