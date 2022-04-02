# Script to train machine learning model.

# Add the necessary imports for the starter code.
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference, slice_performance
from ml.data import process_data
from dotenv import load_dotenv

load_dotenv()
# Add code to load in the data.
clean_data_path = os.environ['CLEAN_DATA_PATH']
model_path = os.environ['MODEL_PATH']
encoder_path = os.environ['ONEHOT_ENCODER_PATH']
lb_path = os.environ['LABEL_ENCODER']

data = pd.read_csv(clean_data_path)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"Precision = {precision} \nRecall = {recall}\nFbeta = {fbeta}")


X_slice, y_slice, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

slices_df = slice_performance(data, X_slice, y_slice, model, cat_features)
with open("slice_output.txt", 'w') as file:
    print(slices_df.to_string(), file=file)


with open(model_path, "wb") as file:
    pickle.dump(model, file)

with open(encoder_path, "wb") as file:
    pickle.dump(encoder, file)

with open(lb_path, "wb") as file:
    pickle.dump(lb, file)
