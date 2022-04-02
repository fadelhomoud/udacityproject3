# Put the code for your API here.

import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
#from ml.data import process_data
from dotenv import load_dotenv

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

load_dotenv()
categorical_features = os.environ['CATEGORICAL_FEATURES'].split(',')
model_path = os.environ['MODEL_PATH']
encoder_path = os.environ['ONEHOT_ENCODER_PATH']
lb_path = os.environ['LABEL_ENCODER']


# Loading model and encoders:
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(encoder_path, "rb") as file:
    encoder = pickle.load(file)

with open(lb_path, "rb") as file:
    lb = pickle.load(file)


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
app = FastAPI()


class PostBody(BaseModel):

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                'age': 27,
                'workclass': 'Private',
                'fnlwgt': 123456,
                'education': 'HS-grad',
                'education-num': 7,
                'marital-status': 'Never-married',
                'occupation': 'Other-service',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Female',
                'capital-gain': 100000,
                'capital-loss': 5000,
                'hours-per-week': 45,
                'native-country': 'England'
            }
        }


@app.get("/")
async def return_greetings():
    return "Welcome to the inference API!"


@app.post("/inference")
async def model_inference(data: PostBody):

    data = data.dict(by_alias=True)

    data, y, encoder_ret, lb_ret = process_data(
        pd.DataFrame(data, index=[0]),
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)

    pred = model.predict(data)
    pred = lb.inverse_transform(pred)

    return {"prediction": pred.tolist()}
