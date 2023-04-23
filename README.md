# Wine quality analysis Repository

## Introduction

This mini-project for SC1015 (Introduction to Data Science and Artificial Intelligence) focusses on the [Wine Quality dataset] (https://www.kaggle.com/datasets/rajyellow46/wine-quality). Our source code can be found [here](https://github.com/reeethong/SC1015Project/blob/main/%5BFINAL%20SUBMISSION%5D%20SC1015%20Wine%20Mini%20Project%20.ipynb).

1. Dataset Preview & EDA
2. Data Cleaning
3. Decision Tree
4. LightGBM Model

## Contributors 
- Reyan - Dataset Preview, Exploratory Data Analysis 
- Ryan - Data Cleaning (Isolation forest), Grid Search
- Yuhan -  Decision Tree Model, Light GBM Model, Findings and Conclusions

## Problem Definition
As avid wine lovers, we like to sample wines that we have not tried before so that we can discover new wines that match our taste but in doing so, we have sampled so many disappointing wines. Hence, we would like to identify the factors that contribute to wine quality so that winemakers can produce wines that are more likely to satisfy consumer preferences. Thus as consumers, we will be more likely to discover wines that we enjoy and are willing to purchase again in the future.


In this project, we will identify which factors are most significant in deciding the quality of red and white wine.

## Models used
### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm that detects anomalies by partitioning data recursively using random splits.  It is easier to isolate an anomaly in a tree because it requires fewer splits, while normal points take more splits to isolate. The algorithm creates isolation trees by randomly selecting a feature and a split value for each node until a termination condition is met. Each point is then given an anomaly score based on how many splits are required to isolate it. By generating an isolation forest for both red and white wine, we are able to identify the data points more likely to be an anomaly and remove them, allowing us for more accurate analysis.

### Decision Trees

The decision tree algorithm selects the feature that splits the data in a way that maximizes the separation between the classes or minimizes the impurity in the target variable. Hence, by examining the decision tree structure, we can identify which features are most frequently used in the tree and at what depth they are used. The features used closer to the root node are generally more important in terms of their ability to predict the target variable. The importance of a feature can also be calculated as the total reduction of error that resulted from splitting that feature.

### Grid Search

To improve our decision tree, we applied grid search to our model. Grid search finds the best combination of hyperparameters for our model. It creates a grid of all the possible combinations of hyperparameters and evaluates each combination using cross-validation. Finally, it would select the combination with the best performance. This technique helped us determine the best hyperparameters to use in our decision trees such as the depth of the tree, minimum samples for leaf nodes and the criterion.

### LightGBM Model

In this project, we employed the LightGBM model, which is a gradient boosting framework utilizing a tree-based learning algorithm. We utilized the Stratified K-Fold cross-validator, employing 5 folds to divide the training data into multiple subsets. Each iteration of our process involved training the model on 4 folds and evaluating its performance on the remaining fold. This entire process was repeated 5 times, with each fold serving as the validation set once.
We proceed by training the LightGBM model for each iteration and using the 'feature_importance()' method to compute the importance score of each physicochemical variable. We then store the computed importance scores of each feature in a dataframe.
Finally, we evaluate our model performance using the root mean squared error (RMSE) metric. Additionally, we plot the feature importance of each variable to gain further insights into the behavior of our model.

## Findings and insights gained
### White wine:
Decision tree model determined that alcohol, volatile acidity, free sulfur dioxide, and chlorides as the most important features for predicting white wine quality. 
Light GBM model indicated that volatile acidity, free sulfur dioxide and chlorides received high feature importance scores.

### Red wine:
Alcohol, sulphates, volatile acidity, and total sulfur dioxide were important features according to the decision tree model. 
Light GBM model identified a different set of top features, namely, total sulfur dioxide, density, residual sugar, and volatile acidity. 
Only two factors, total sulfur dioxide and volatile acidity, overlapped between the two models. Alcohol and sulphates were not even in the top 8 most important features in Light GBM model. 

This inconsistency may be due to the decision tree model being adept at capturing non-linear relationships between features, while Light GBM may not be as effective in this regard. Additionally, the decision tree model is susceptible to noise in the dataset, whereas Light GBM is more robust and better able to ignore irrelevant features.


## Conclusion
- Red and White wines have very distinct differences in their distribution of physiochemical makeups and should be analysed separately.
- To improve white wine quality, the wine makers should exercise greater control over 3 main factors, by decreasing volatile acidity, free sulfur dioxide, and chlorides.
- Red wine makers should focus on the factors of total sulfur dioxide and volatile acidity in order to improve wine quality. They should simultaneously decrease total sulfur dioxide levels and increase volatile acidity.

## Our learning points 
- Data binning and separating data for analysis
- Anomaly detection: Isolation forest
- Other packages such as Light GBM
- Machine Learning Models:
  - Decision tree with grid search
  - Light GBM model with stratified K-Fold cross-validation

## Presentation slides
https://www.canva.com/design/DAFfUENcVMs/OiC64Qu3Sc1h5QHomGXG9w/edit?utm_content=DAFfUENcVMs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## References
- https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
- https://www.kaggle.com/code/mgmarques/wines-type-and-quality-classification-exercises
- https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
- https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/
