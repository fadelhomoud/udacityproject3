from dotenv import load_dotenv
import os
load_dotenv()
catt = os.environ['CATEGORICAL_FEATURES'].split(',')
print(catt)

