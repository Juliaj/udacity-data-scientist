airbnb
==============================

Comparing to hotels, Airbnb provides travelers more cost effective options with greater flexibility. To property owners, Airbnb opens up a new world to reach to many customers with an easy access to its platform. In this repo, we examine Seattle and Boston Airbnb data following CRISP-DM methodology and draw insights.

Installation
------------
Running Jupyter notebooks require Python 3.8 or above. For additional  dependency, run following command:
```
$ pip install -r requirements.txt
```

Data Analysis and Rental Price Prediction
------------
Data preparation and analysis are stored in various Jupyter notebooks:
- **Seattle Data Preparation.ipynb** and **Boston Data Preparation.ipynb** list the steps to impute missing data, process categorical fields and convert object type field such as price to numerica field. 

- **Airbnb Seattle versus Boston.ipynb** analyzes the similarity and difference of the airbnb rentals in Seattle and Boston along following angles:

    - top 10 neighbourhood for listing rentals
    - number of active rentals by room type
    - luxary rentals and long term rentals
    - host response rate and review ratings

- **Airbnb Rental Price Prediction .ipynb** details the modeling of rental price. Two models are produced based on different handling of property_type feature. 

Blog 
------------
https://github.com/Juliaj/udacity-data-scientist/blob/gh-pages/index.md 

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── data           
       │    └── listings.py <- module to process rental listings
       │    └── rental_calendar.py <- module to process calendars 
       │    └── reviews.py <- module to process reviews 
       │
       ├── features       
       │   └── price.py <- feature extraction for price modeling
       │
       └── models         
           └── predict_price.py <- linear regression model and cutoff search
    


Acknowledgement 
------------
The Airbnb dataset for Seattle and Boston are provided by Kaggle. 
- Boston data: https://www.kaggle.com/airbnb/boston/data
- Seattle data: https://www.kaggle.com/airbnb/seattle/data  

Project was created based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
