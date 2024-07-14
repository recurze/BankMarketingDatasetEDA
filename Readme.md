## Goals

* Exploratory data analysis (EDA) of the Portugese Bank Marketing dataset ([UCI][1], [kaggle][2]).
* Classification goal of predicting whether or not a customer will subscribe a term deposit.

## Info

* This was a group project part of the course "DSA5101: Introduction to Big Data for Industry". I did the EDA and pre-processing bits.
* I've also included some models to demo performance. A more thorough study of different models and parameters was performed by the other members.
* The presentation also included discussions of the business implication, insights and findings, and future ideas. I'll not be updating the slides here.
* Please download the dataset from the links provided. I'll not be uploading the csv files here.

[1]: https://archive.ics.uci.edu/dataset/222/bank+marketing
[2]: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

## Summary

### EDA

* Get metadata about the data frame (like number of entries) and its columns (like names and type).
* Distinguish between numerical and categorical columns and treat them separately.
* Get the distribution of the output. Here, it's imbalanced classification (a.k.a outlier/novelty detection).
* Get the distributions of the individual columns and any relevant conditional distributions (e.g.: column | output = 1).
* Impute unknown/null values.
* Check for (and remove) outliers.
* Drop columns that seem logically irrelevant (carefully).
* Create new features.
* Find correlations using mutual information and correlation matrices.

### Preprocessing

* Use pipelines to streamline processing.
* Transform the data:
    - Incorporate adding/dropping columns from EDA into pipelines and leave the original data untouched.
    - Use column transformers to scale numerical columns and encode + scale categorical columns.
    - If your model requires it, consider normalizing.
* Select features:
    - Remove noisy features to avoid overfitting.
    - Remove interdependent features.
    - Use dimensionality reduction, univariate feature selection or select from model.
* Consider oversampling to tackle the imbalanced dataset.

The pipeline looks like this:
```python
model = IMBPipeline([
    ('cleaner', data_cleaner),  # to drop columns and imputation
    ('feature_engineer', feature_engineer),  # to add new columns
    ('column_transformer', column_transformer),
    ('feature_selector', 'passthrough'),
    ('oversampler', SMOTE(random_state=0)),
    ('predictor', RandomForestClassifier(random_state=0)),
])
```

### Models

* Trees:
    - Boosting
    - Bagging
    - Balanced
    - Ensemble
* Linear and NNs

### Performance Analysis

* Scoring
    - Choose scoring metric (more than one preferred).
    - Cross validate or simple train test split.
    - Generate report (e.g., confusion_matrix).
* Model Selection:
    - Use grid search for the best hyper parameters.
    - Use tuner if you are using deep learning.
