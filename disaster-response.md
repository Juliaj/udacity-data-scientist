## Training a classifier for Disaster Response Messages

In recent years, climate changes have increased the chances for severe weather and forest fires. When a natural disaster strikes, the ability of filtering through the flood of communications and classifying them properly is critical for disaster response organizations to provide timely reliefs. 

![Image](./images/disaster_response/lerone-pieters-93gXyV16hZs-unsplash.jpg)
https://unsplash.com/photos/93gXyV16hZs?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink

To automate such tasks, machine learning has shown greater advantage comparing to imperative software development due to its ability of extracting patterns directly from data. 

In this article, we discuss the ETL and ML training piplelines for this type of message classification task and findings with the model quality.

### Data Processing

The data collection from [Figure Eight](https://www.figure-eight.com/) consists of two csv files: one with raw text messages and the other having the corresponding categories. The ETL pipeline follows the general practice of CRISP, the cleaned data are saved into a sqlite database. A special note about the `categories` which will become the labels for subsequete classification task.

- The `categories` column from csv file is splitted into 36 seperate columns. The new columns map the message to needs such as 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people' etc. Most of the categories have value of 0 or 1 except for `related` column, some rows have value of `2`. Instead of dropping the rows, the value of `2` has been replaced by `1`. 

There are 26,386 raw messages. At end of ETL pipeline, 26,216 rows remaining.  

### Model Training

The training task is a multi-label classification which support a scenario that a single message can be mapped to mutiple categories. 

The feature extraction utilizes a TF-idf transformer in conjuction with a custom transformer to identify whether a sentence starting with a verb. The implementaion involves MultiOutputClassifier from sklearn with KNeighborsClassifier as estimator. 

Hyper-parameter tuning are done via GridSearchCV with two parameters: number of the clusters and ngram_range. 

### Evaluate Model Quality

For multi-label classification, the metrics from the classification_report such as f1_score, precision, recall are generally used for evaluation. 

Explained in [Precision vs Recall](https://medium.com/@shrutisaxena0617/precision-vs-recall-386cf9f89488), "precision" measures the percentage of the results which are relevant. On the other hand, "recall" refers to the percentage of total relevant results correctly classified by the algorithm. Recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”. 

Another good reference for the metrics is [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf).  

Generally trade-offs have to be made to use any of these metric to fine tune the algorithm and hyper-parameters. In our case, it is a binary classification for each category, here are high level findings of these metrics.

- For the positives, most of the precision scores are greater than .50 whereas the recall scores are generally low. This most likely are due to the imbalanced dataset where the support for the 'positives' is low. 
- It may be challenging to use the weighted avg of f1-score to evaluate model performance across labels. The weighted avg of f1-score for `request` is higher than `related`, this may indicates the model fairs a bit better for classfying `request`. However, this is not consistent, meaning, higher value doesn't directly correlate to precision nor recall metrics for that category (e.g. `security`).    
- Number of categories such as `food`, `shelter` and `clothing` etc, metrics show the model perform reasonably well. 

```
category = related
              precision    recall  f1-score   support

           0       0.59      0.22      0.32      1873
           1       0.80      0.95      0.87      5992

weighted avg       0.75      0.78      0.74      7865

category = request
              precision    recall  f1-score   support

           0       0.85      0.99      0.92      6533
           1       0.78      0.13      0.23      1332

weighted avg       0.84      0.85      0.80      7865

category = offer
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7829
           1       1.00      0.00      0.00        36

weighted avg       1.00      1.00      0.99      7865

category = aid_related
              precision    recall  f1-score   support

           0       0.60      0.99      0.75      4646
           1       0.79      0.07      0.12      3219

weighted avg       0.68      0.61      0.49      7865

category = medical_help
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      7227
           1       0.43      0.01      0.02       638

weighted avg       0.88      0.92      0.88      7865

category = medical_products
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      7447
           1       0.62      0.02      0.05       418

weighted avg       0.93      0.95      0.92      7865

category = search_and_rescue
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      7673
           1       1.00      0.00      0.00       192

weighted avg       0.98      0.98      0.96      7865

category = security
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      7721
           1       0.00      0.00      0.00       144

weighted avg       0.96      0.98      0.97      7865

category = military
              precision    recall  f1-score   support

           0       0.97      1.00      0.98      7620
           1       1.00      0.00      0.00       245

weighted avg       0.97      0.97      0.95      7865

category = child_alone
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7865

weighted avg       1.00      1.00      1.00      7865

category = water
              precision    recall  f1-score   support

           0       0.94      1.00      0.97      7365
           1       0.51      0.05      0.09       500

weighted avg       0.91      0.94      0.91      7865

category = food
              precision    recall  f1-score   support

           0       0.90      0.99      0.94      6987
           1       0.70      0.10      0.17       878

weighted avg       0.88      0.89      0.86      7865

category = shelter
              precision    recall  f1-score   support

           0       0.92      1.00      0.95      7160
           1       0.70      0.06      0.11       705

weighted avg       0.90      0.91      0.88      7865

category = clothing
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      7750
           1       0.62      0.04      0.08       115

weighted avg       0.98      0.99      0.98      7865

category = money
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      7695
           1       0.33      0.01      0.01       170

weighted avg       0.96      0.98      0.97      7865

category = missing_people
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      7773
           1       0.00      0.00      0.00        92

weighted avg       0.98      0.99      0.98      7865

category = refugees
              precision    recall  f1-score   support

           0       0.97      1.00      0.98      7605
           1       0.00      0.00      0.00       260

weighted avg       0.93      0.97      0.95      7865

category = death
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      7499
           1       0.52      0.04      0.08       366

weighted avg       0.93      0.95      0.93      7865

category = other_aid
              precision    recall  f1-score   support

           0       0.87      1.00      0.93      6832
           1       0.45      0.01      0.03      1033

weighted avg       0.82      0.87      0.81      7865

category = infrastructure_related
              precision    recall  f1-score   support

           0       0.94      1.00      0.97      7360
           1       0.00      0.00      0.00       505

weighted avg       0.88      0.94      0.90      7865

category = transport
              precision    recall  f1-score   support

           0       0.95      1.00      0.98      7503
           1       0.00      0.00      0.00       362

weighted avg       0.91      0.95      0.93      7865

category = buildings
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      7473
           1       0.59      0.03      0.05       392

weighted avg       0.93      0.95      0.93      7865

category = electricity
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      7697
           1       0.20      0.01      0.01       168

weighted avg       0.96      0.98      0.97      7865

category = tools
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      7817
           1       1.00      0.00      0.00        48

weighted avg       0.99      0.99      0.99      7865

category = hospitals
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      7787
           1       1.00      0.00      0.00        78

weighted avg       0.99      0.99      0.99      7865

category = shops
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7837
           1       1.00      0.00      0.00        28

weighted avg       1.00      1.00      0.99      7865

category = aid_centers
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      7762
           1       1.00      0.00      0.00       103

weighted avg       0.99      0.99      0.98      7865

category = other_infrastructure
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      7524
           1       0.00      0.00      0.00       341

weighted avg       0.92      0.96      0.94      7865

category = weather_related
              precision    recall  f1-score   support

           0       0.74      0.99      0.85      5702
           1       0.74      0.07      0.13      2163

weighted avg       0.74      0.74      0.65      7865

category = floods
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      7242
           1       0.68      0.03      0.05       623

weighted avg       0.90      0.92      0.89      7865

category = storm
              precision    recall  f1-score   support

           0       0.91      1.00      0.95      7127
           1       0.62      0.05      0.09       738

weighted avg       0.88      0.91      0.87      7865

category = fire
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      7782
           1       0.00      0.00      0.00        83

weighted avg       0.98      0.99      0.98      7865

category = earthquake
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      7163
           1       0.71      0.12      0.20       702

weighted avg       0.90      0.92      0.89      7865

category = cold
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      7694
           1       1.00      0.01      0.01       171

weighted avg       0.98      0.98      0.97      7865

category = other_weather
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      7450
           1       0.12      0.00      0.00       415

weighted avg       0.90      0.95      0.92      7865

category = direct_report
              precision    recall  f1-score   support

           0       0.82      0.99      0.90      6321
           1       0.74      0.10      0.17      1544

weighted avg       0.80      0.82      0.75      7865

```
#### A note about other `score` metrics 
-  What is the `score` logged by GridSearchCV? In each iteration, this score is calculated in order to reduce the overfitting. A high validation score reflects a more generalized model .

- Can we use the output from Pipeline `Score` method for model evalaution? This score is only useful for overall model training and validation and it isn't granular enough to evaluate for each label.

### Future development

Both the precision and recall scores of existing models can be improved. There are two areas that deserve consideration, 

- Handle imbalanced dataset especially for categories like water which have very limited positive data samples. A few techniques were discussed in this [post](https://medium.com/james-blogs/handling-imbalanced-data-in-classification-problems-7de598c1059f).

- Add additional featuers, such as including data from `genre`. The current model is trained on messages.

### Acknowledgement 
The training data are from [Figure Eight](https://www.figure-eight.com/) and made available by Udacity [Data Scientist Nanodegree program](https://classroom.udacity.com/nanodegrees/nd025). 


