"""
A script to test the deployed app on heroku.
"""

import requests

data = {
        'age': 27,
        'workclass': 'Private',
        'fnlwgt': 123456,
        'education': 'Masters',
        'education-num': 7,
        'marital-status': 'Never-married',
        'occupation': 'Other-service',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 100000,
        'capital-loss': 5000,
        'hours-per-week': 45,
        'native-country': 'England'
    }

r = requests.post('https://predict-income-mlops.herokuapp.com/inference', json=data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")