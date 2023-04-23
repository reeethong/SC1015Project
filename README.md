# Wine quality analysis Repository

## Introduction

This mini-project for SC1015 (Introduction to Data Science and Artificial Intelligence) focusses on the Wine Quality dataset (https://www.kaggle.com/datasets/rajyellow46/wine-quality). Our source code can be found in the following hyperlinks.

1. [Dataset Preview & EDA](https://github.com/reeethong/SC1015Project/blob/main/%5BFINAL%20SUBMISSION%5D%20SC1015%20Wine%20Mini%20Project%20.ipynb)
2. [Data Cleaning]
3. [Decision Tree]
4. [LightGBM Model]

1. [Data Extraction](https://github.com/nicklimmm/movie-analysis/blob/main/data-extraction.ipynb)
2. [Data Visualization](https://github.com/nicklimmm/movie-analysis/blob/main/data-visualization.ipynb)
3. [Data Resampling and Splitting](https://github.com/nicklimmm/movie-analysis/blob/main/data-resampling-and-splitting.ipynb)
4. [Logistic Regression](https://github.com/nicklimmm/movie-analysis/blob/main/logistic-regression.ipynb)
5. [Neural Network](https://github.com/nicklimmm/movie-analysis/blob/main/neural-network.ipynb)

# Contributors 
- Reyan - Dataset Preview, Exploratory Data Analysis 
- Ryan - Data Cleaning (Isolation forest), Grid Search
- Yuhan -  Decision Tree Model, Light GBM Model, Conclusions

# Problem Definition
## As university students, we are new to the wine scene and would like to enjoy wine as much as our parents. Even within red and white wine, there are so many different wine products available on the market. Hence, we would like to find out which factors are important in predicting wine quality so that we can make more informed decisions when purchasing wine and choose wines that are more likely to match our preferences. 

# Models used
Isolation Forest
Isolation Forest is an unsupervised machine learning algorithm that detects anomalies by partitioning data recursively using random splits.  It is easier to isolate an anomaly in a tree because it requires fewer splits, while normal points take more splits to isolate. The algorithm creates isolation trees by randomly selecting a feature and a split value for each node until a termination condition is met. Each point is then given an anomaly score based on how many splits are required to isolate it. By generating an isolation forest for both red and white wine, we are able to identify the data points more likely to be an anomaly and remove them, allowing us for more accurate analysis.
Decision Trees
The decision tree algorithm selects the feature that splits the data in a way that maximizes the separation between the classes or minimizes the impurity in the target variable. Hence, by examining the decision tree structure, we can identify which features are most frequently used in the tree and at what depth they are used. The features used closer to the root node are generally more important in terms of their ability to predict the target variable. The importance of a feature can also be calculated as the total reduction of error that resulted from splitting that feature.
Grid Search
To improve our decision tree, we applied grid search to our model. Grid search finds the best combination of hyperparameters for our model. It creates a grid of all the possible combinations of hyperparameters and evaluates each combination using cross-validation. Finally, it would select the combination with the best performance. This technique helped us determine the best hyperparameters to use in our decision trees such as the depth of the tree, minimum samples for leaf nodes and the criterion.
LightGBM Model
In this project, we employ the LightGBM model, which is a gradient boosting framework utilizing a tree-based learning algorithm. We utilize the Stratified K-Fold cross-validator, employing 5 folds to divide the training data into multiple subsets. Each iteration of our process involves training the model on 4 folds and evaluating its performance on the remaining fold. This entire process is repeated 5 times, with each fold serving as the validation set once.
We proceed by training the LightGBM model for each iteration and using the 'feature_importance()' method to compute the importance score of each physicochemical variable. We then store the computed importance scores of each feature in a dataframe.
Finally, we evaluate our model performance using the root mean squared error (RMSE) metric. Additionally, we plot the feature importance of each variable to gain further insights into the behavior of our model.

# Findings and insights gained
White wine:
Decision tree model determined that alcohol, volatile acidity, free sulfur dioxide, and chlorides as the most important features for predicting white wine quality. 
Light GBM model indicated that volatile acidity, free sulfur dioxide, and chlorides got high feature importance scores.  
Red wine:
Alcohol, sulphates, volatile acidity, and total sulfur dioxide were important features according to the decision tree model. 
Light GBM model identified a different set of top features, namely, total sulfur dioxide, density, residual sugar, and volatile acidity. 
Only two factors – total sulfur dioxide and volatile acidity overlapped between the two models. Alcohol and sulphates are not even in the top8 most important features in Light GBM model. 
This inconsistency may be due to the decision tree model being adept at capturing non-linear relationships between features, while Light GBM may not be as effective in this regard. Additionally, the decision tree model is susceptible to noise in the dataset, whereas Light GBM is more robust and better able to ignore irrelevant features.


# Conclusion
Red and White wine have very distinct differences in their distribution of physiochemical makeups and should be considered separately.
To improve white wine quality, the wine makers should exercise greater control over these four factors by decreasing volatile acidity, free sulfur dioxide, and chlorides.
The wine makers should prioritize their attention on the factors of total sulfur dioxide and volatile acidity in order to improve red wine quality. This can be achieved by simultaneously increasing volatile acidity and decreasing total sulfur dioxide levels.

# Our learning points 
Identifying trends and separating data for analysis
Anomaly detection: Isolation forest
Other packages such as Light GBM
Machine Learning Models:
Decision tree with grid search
Light GBM model with stratified K-Fold cross-validation

# References
https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/
https://www.canva.com/design/DAFfUENcVMs/OiC64Qu3Sc1h5QHomGXG9w/edit?utm_content=DAFfUENcVMs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
