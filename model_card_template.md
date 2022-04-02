# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Random Forest Classfier from Scikit-Learn library in Python created by Fadel Homoud. It is created for the 3rd project of Udacity's MLOps Nano-Degree. It uses the default parameters except for the `random_state` being 221.

## Intended Use
The model is built to predict if a USA citizen is making more or less than $50k per year given some data about the person like the age, race, gender and other characteristics.

## Training Data
The data was downloaded from (https://archive.ics.uci.edu/ml/datasets/census+income). The dataset contains 32561 observation, and the model was trained on a random sample of the data. The random sample is 80% of the data. All categorical variables were One-Hot encoded using `OneHotEncoder` from Scikit-Learn. The label column "salary" was also transformed to be the either a >50k or <50k using `LabelBinarizer` from Scikit-Learn

## Evaluation Data
The model was evaluated on the 20% of the data that were holded for testing only.

## Metrics
The metrics that were considered and calculated on the testing set are as follow:
- Precision = 0.7343635025754232 
- Recall = 0.6418006430868167
- Fbeta = 0.6849691146190804

## Ethical Considerations
Looking at the model performance on the slices of the data, we can see that some groups have lower performance than others. For example, for Native Country = South, the precision = 0.88 while most others native countries are above 0.9. Therefore, this is caused by the imbalance in the data, therefore the model should be used carefully.

## Caveats and Recommendations
The data used to train the model is very old and probably makes it useless for predicting the income for the current time. The data was collected 1994, and so many things has changed in the last 28 years.  
It is recommened to use new data to train the model if it is to be used for predictin the income of people from this time.