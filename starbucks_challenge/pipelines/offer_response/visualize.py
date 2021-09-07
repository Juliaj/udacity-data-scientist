"""Charts for offer response binary classification.

Code is adapted from https://github.com/LailaSabar/Starbucks-Capstone-Challenge/blob/master/Starbucks_Capstone_notebook.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def plot_feature_importance(feature_imp):
    """Plot the top 10 important features
    """
    fig, ax = plt.subplots(figsize=(10, 10),nrows=1,ncols=1)
    sns.barplot(x=feature_imp.loc[0:10,'imp_perc'], y=feature_imp.loc[0:10,'feature'], color='#95DAC1', data=feature_imp)
    plt.xlabel('Percentage')
    plt.ylabel('Feature')
    plt.title('Top 10 important features')
    plt.show()


def plot_confusion_matrix(conf_matrix, unique_classes, normalized=False, cmap=plt.cm.Greys):
    """Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Args:
        conf_matrix: confusion matrix calculated. 
        unique_classes: (nd.array) unique labels from the target
        normalize: (boolean) used in chart title
            
    Returns:
     ax: matplotlib axes object.

    """
    np.set_printoptions(precision=2)

    if normalized:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=unique_classes, yticklabels=unique_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalized else 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
