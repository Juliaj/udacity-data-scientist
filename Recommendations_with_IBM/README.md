# Recommendation For IBM Watson Studio Users

IBM Watson Studio is a great platform for users who are passionate with Data Science to learn, interact and build projects. When logging in, user sees a dashboard showing various articles. In this project, we explore ways to provide a recommendation board that shows the articles that are most pertinent to a specific user.

### Instructions

1. Install Python dependencies.
```
$ pip install -r requirements.txt
```

2. Open the Recommendations_with_IBM.ipynb in Jupyter notebook.

Project Organization
------------
    ├── data           
    │   └── articles_community.csv <- article title, body and descriptions. 
    │   └── user-item-interactions.csv <- user interaction data with articles.
    │
    ├── Recommendations_with_IBM.ipynb <- implementation of various recommenders.          
    │   └── user_item_matrix.p <- pickle file for a matrix capturing user and article interaction.
    │   └── top_x.p <- pickle file for top x aritcles.
    │
    ├── Recommendations_with_IBM.html  <- The html version of Jupyter notebook.
    │
    ├── README.md          <- The top-level README for developers using this project.
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
    
### Recommendations
Following methods were explored in this project:

- Rank Based Recommendations.
- User-User Based Collaborative Filtering. 
- Matrix Factorization.

Content Based Recommendations will be explored in the future.

### Acknowledgement 
The data in this project are made avialable by Udacity [Data Scientist Nanodegree program](https://classroom.udacity.com/nanodegrees/nd025). 
