import pandas as pd
data = pd.read_csv('census.csv')

data.columns = data.columns.str.strip()
data.columns
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.strip()

data.to_csv('cleaned_census.csv', index=False)